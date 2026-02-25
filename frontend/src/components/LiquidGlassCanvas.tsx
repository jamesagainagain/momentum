import { useRef, useEffect, useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { MultiPassRenderer, loadTextureFromURL } from "@/lib/liquid-glass/GLUtils";
import { computeGaussianKernelByRadius } from "@/lib/liquid-glass/utils";

import VertexShader from "@/lib/liquid-glass/shaders/vertex.glsl?raw";
import FragmentBgShader from "@/lib/liquid-glass/shaders/fragment-bg.glsl?raw";
import FragmentBgVblurShader from "@/lib/liquid-glass/shaders/fragment-bg-vblur.glsl?raw";
import FragmentBgHblurShader from "@/lib/liquid-glass/shaders/fragment-bg-hblur.glsl?raw";
import FragmentMainShader from "@/lib/liquid-glass/shaders/fragment-main.glsl?raw";

export interface LiquidGlassCanvasProps {
  /** Width in pixels. */
  width: number;
  /** Height in pixels. */
  height: number;
  className?: string;
  /** Optional image URL for background. If not set, a solid background color is used. */
  backgroundTextureUrl?: string | null;
  /** Blur radius for glass (default 24). */
  blurRadius?: number;
  /** Shape roundness / superellipse exponent (default 4). */
  shapeRoundness?: number;
  /** Enable mouse-reactive glass (default true). Respects prefers-reduced-motion. */
  interactive?: boolean;
}

const DEFAULT_BLUR = 24;
const DEFAULT_ROUNDNESS = 4;

function createSolidColorTexture(gl: WebGL2RenderingContext, hex: string): WebGLTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 1;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d context");
  ctx.fillStyle = hex;
  ctx.fillRect(0, 0, 1, 1);

  const texture = gl.createTexture();
  if (!texture) throw new Error("Failed to create texture");
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return texture;
}

export function LiquidGlassCanvas({
  width,
  height,
  className,
  backgroundTextureUrl = null,
  blurRadius = DEFAULT_BLUR,
  shapeRoundness = DEFAULT_ROUNDNESS,
  interactive = true,
}: LiquidGlassCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [webglUnavailable, setWebglUnavailable] = useState(false);
  const pointerRef = useRef({ x: width / 2, y: height / 2 });
  const prefersReducedMotion =
    typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const dpr = typeof window !== "undefined" ? Math.min(window.devicePixelRatio, 2) : 1;
  const canvasWidth = Math.round(width * dpr);
  const canvasHeight = Math.round(height * dpr);

  const shapeW = (width * 0.85 * dpr) / 2;
  const shapeH = (height * 0.7 * dpr) / 2;

  const render = useCallback((): (() => void) | void => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2");
    if (!gl) {
      setWebglUnavailable(true);
      return;
    }

    const ext = gl.getExtension("EXT_color_buffer_float");
    if (!ext) {
      setWebglUnavailable(true);
      return;
    }

    const renderer = new MultiPassRenderer(canvas, [
      {
        name: "bgPass",
        shader: { vertex: VertexShader, fragment: FragmentBgShader },
      },
      {
        name: "vBlurPass",
        shader: { vertex: VertexShader, fragment: FragmentBgVblurShader },
        inputs: { u_prevPassTexture: "bgPass" },
      },
      {
        name: "hBlurPass",
        shader: { vertex: VertexShader, fragment: FragmentBgHblurShader },
        inputs: { u_prevPassTexture: "vBlurPass" },
      },
      {
        name: "mainPass",
        shader: { vertex: VertexShader, fragment: FragmentMainShader },
        inputs: { u_blurredBg: "hBlurPass", u_bg: "bgPass" },
        outputToScreen: true,
      },
    ]);

    const blurWeights = computeGaussianKernelByRadius(blurRadius);

    const bgRef = { texture: null as WebGLTexture | null, ratio: 1 };
    if (backgroundTextureUrl) {
      loadTextureFromURL(gl, backgroundTextureUrl).then(
        (r) => {
          bgRef.texture = r.texture;
          bgRef.ratio = r.ratio;
        },
        () => {
          bgRef.texture = createSolidColorTexture(gl, "#faf9f5");
          bgRef.ratio = 1;
        }
      );
    } else {
      bgRef.texture = createSolidColorTexture(gl, "#faf9f5");
      bgRef.ratio = 1;
    }

    let rafId: number;
    let disposed = false;

    const loop = () => {
      if (disposed) return;
      rafId = requestAnimationFrame(loop);

      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      const mouseX = pointerRef.current.x * dpr;
      const mouseY = (height - pointerRef.current.y) * dpr;
      const shapeSizeSpringX = shapeW;
      const shapeSizeSpringY = shapeH;
      const shapeRadiusPx = (Math.min(shapeSizeSpringX, shapeSizeSpringY) / 2) * 0.5;

      renderer.setUniforms({
        u_resolution: [canvasWidth, canvasHeight],
        u_dpr: dpr,
        u_blurWeights: blurWeights,
        u_blurRadius: blurRadius,
        u_mouse: [mouseX, mouseY],
        u_mouseSpring: [mouseX, mouseY],
        u_shapeWidth: shapeSizeSpringX,
        u_shapeHeight: shapeSizeSpringY,
        u_shapeRadius: shapeRadiusPx,
        u_shapeRoundness: shapeRoundness,
        u_mergeRate: 0.15,
        u_glareAngle: (45 * Math.PI) / 180,
        u_showShape1: 0,
      });

      const hasBg = bgRef.texture != null;
      renderer.render({
        bgPass: {
          u_bgType: hasBg ? 3 : 0,
          u_bgTexture: hasBg ? bgRef.texture! : undefined,
          u_bgTextureRatio: bgRef.ratio,
          u_bgTextureReady: hasBg ? 1 : 0,
          u_shadowExpand: 80,
          u_shadowFactor: 0.5,
          u_shadowPosition: [0, 0],
        },
        mainPass: {
          u_tint: [1, 1, 1, 0.08],
          u_refThickness: 80,
          u_refFactor: 1.5,
          u_refDispersion: 0.1,
          u_refFresnelRange: 500,
          u_refFresnelHardness: 0.5,
          u_refFresnelFactor: 0.5,
          u_glareRange: 400,
          u_glareHardness: 0.5,
          u_glareConvergence: 0.5,
          u_glareOppositeFactor: 0.5,
          u_glareFactor: 0.15,
          u_blurEdge: 0,
          STEP: 9,
        },
      });
    };

    rafId = requestAnimationFrame(loop);

    return () => {
      disposed = true;
      cancelAnimationFrame(rafId);
      if (bgRef.texture) gl.deleteTexture(bgRef.texture);
      renderer.dispose();
    };
  }, [
    width,
    height,
    canvasWidth,
    canvasHeight,
    dpr,
    shapeW,
    shapeH,
    blurRadius,
    shapeRoundness,
    backgroundTextureUrl,
  ]);

  useEffect(() => {
    const cleanup = render();
    return () => {
      if (typeof cleanup === "function") cleanup();
    };
  }, [render]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || webglUnavailable) return;

    const onPointerMove = (e: PointerEvent) => {
      if (!interactive || prefersReducedMotion) return;
      const rect = canvas.getBoundingClientRect();
      pointerRef.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    };

    canvas.addEventListener("pointermove", onPointerMove);
    return () => canvas.removeEventListener("pointermove", onPointerMove);
  }, [interactive, prefersReducedMotion, webglUnavailable]);

  if (webglUnavailable) {
    return (
      <div
        className={cn("glass-panel rounded-lg", className)}
        style={{ width, height }}
        aria-hidden
      />
    );
  }

  return (
    <canvas
      ref={canvasRef}
      width={canvasWidth}
      height={canvasHeight}
      className={cn("block rounded-lg", className)}
      style={{ width, height }}
      aria-hidden
    />
  );
}
