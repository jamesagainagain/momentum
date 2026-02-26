import { useRef, useEffect, useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { MultiPassRenderer, loadTextureFromURL } from "@liquid-glass/utils/GLUtils";
import { computeGaussianKernelByRadius } from "@liquid-glass/utils/index";
import type { LiquidGlassPresetControls } from "@/lib/liquid-glass/preset";

import VertexShader from "@liquid-glass/shaders/vertex.glsl?raw";
import FragmentBgShader from "@liquid-glass/shaders/fragment-bg.glsl?raw";
import FragmentBgVblurShader from "@liquid-glass/shaders/fragment-bg-vblur.glsl?raw";
import FragmentBgHblurShader from "@liquid-glass/shaders/fragment-bg-hblur.glsl?raw";
import FragmentMainShader from "@liquid-glass/shaders/fragment-main.glsl?raw";

export interface LiquidGlassCanvasProps {
  width: number;
  height: number;
  className?: string;
  backgroundTextureUrl?: string | null;
  blurRadius?: number;
  shapeRoundness?: number;
  interactive?: boolean;
  /** When true, the glass shape fills the entire canvas (for pill/bar). */
  fillShape?: boolean;
  /** Optional preset from liquid-glass-studio (overrides defaults). */
  preset?: LiquidGlassPresetControls | null;
  /** When true and WebGL is unavailable, render transparent (no fallback bar). */
  fallbackTransparent?: boolean;
  /** When fillShape is false: pill width as ratio of canvas width (0–1). Default 0.85. */
  shapeWidthRatio?: number;
  /** When fillShape is false: pill height as ratio of canvas height (0–1). Default 0.7. */
  shapeHeightRatio?: number;
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
  fillShape = false,
  preset = null,
  fallbackTransparent = false,
  shapeWidthRatio = 0.85,
  shapeHeightRatio = 0.7,
}: LiquidGlassCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [webglUnavailable, setWebglUnavailable] = useState(false);
  const pointerRef = useRef({ x: width / 2, y: height / 2 });
  const prefersReducedMotion =
    typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const dpr = typeof window !== "undefined" ? Math.min(window.devicePixelRatio, 2) : 1;
  const canvasWidth = Math.round(width * dpr);
  const canvasHeight = Math.round(height * dpr);

  const shapeW = fillShape ? canvasWidth : (width * shapeWidthRatio * dpr) / 2;
  const shapeH = fillShape ? canvasHeight : (height * shapeHeightRatio * dpr) / 2;

  const render = useCallback((): (() => void) | void => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2", {
      alpha: true,
      premultipliedAlpha: false,
    });
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

    const blurRadiusUsed = preset?.blurRadius ?? blurRadius;
    const blurWeights = computeGaussianKernelByRadius(blurRadiusUsed);

    const bgRef = { texture: null as WebGLTexture | null, ratio: 1 };
    if (backgroundTextureUrl) {
      loadTextureFromURL(gl, backgroundTextureUrl).then(
        (r) => {
          bgRef.texture = r.texture;
          bgRef.ratio = r.ratio;
        },
        () => {
          bgRef.texture = createSolidColorTexture(gl, "#fbf9f6");
          bgRef.ratio = 1;
        }
      );
    } else {
      bgRef.texture = createSolidColorTexture(gl, "#fbf9f6");
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
      const shapeRadiusPx = fillShape
        ? Math.min(canvasWidth, canvasHeight) / 2
        : (() => {
            const minHalf = Math.min(shapeSizeSpringX, shapeSizeSpringY) / 2;
            const pct = preset?.shapeRadius ?? 50;
            return minHalf * (pct / 100);
          })();

      const mergeRate = preset?.mergeRate ?? 0.15;
      const shapeRoundnessUsed = preset?.shapeRoundness ?? shapeRoundness;
      const glareAngleDeg = preset?.glareAngle ?? 45;
      const showShape1 = preset?.showShape1 ? 1 : 0;

      renderer.setUniforms({
        u_resolution: [canvasWidth, canvasHeight],
        u_dpr: dpr,
        u_blurWeights: blurWeights,
        u_blurRadius: blurRadiusUsed,
        u_mouse: [mouseX, mouseY],
        u_mouseSpring: [mouseX, mouseY],
        u_shapeWidth: shapeSizeSpringX,
        u_shapeHeight: shapeSizeSpringY,
        u_shapeRadius: shapeRadiusPx,
        u_shapeRoundness: shapeRoundnessUsed,
        u_mergeRate: mergeRate,
        u_glareAngle: (glareAngleDeg * Math.PI) / 180,
        u_showShape1: showShape1,
      });

      const t = preset?.tint ?? { r: 255, g: 255, b: 255, a: 8 };
      const shadowPos = preset?.shadowPosition ?? { x: 0, y: 0 };
      const hasBg = bgRef.texture != null;
      renderer.render({
        bgPass: {
          u_bgType: hasBg ? 3 : 0,
          u_bgTexture: hasBg ? bgRef.texture! : undefined,
          u_bgTextureRatio: bgRef.ratio,
          u_bgTextureReady: hasBg ? 1 : 0,
          u_shadowExpand: preset?.shadowExpand ?? 80,
          u_shadowFactor: (preset?.shadowFactor ?? 50) / 100,
          u_shadowPosition: [-shadowPos.x, -shadowPos.y],
        },
        mainPass: {
          u_tint: [t.r / 255, t.g / 255, t.b / 255, t.a / 255],
          u_refThickness: preset?.refThickness ?? 80,
          u_refFactor: preset?.refFactor ?? 1.5,
          u_refDispersion: preset?.refDispersion ?? 0.1,
          u_refFresnelRange: preset?.refFresnelRange ?? 500,
          u_refFresnelHardness: (preset?.refFresnelHardness ?? 50) / 100,
          u_refFresnelFactor: (preset?.refFresnelFactor ?? 50) / 100,
          u_glareRange: preset?.glareRange ?? 400,
          u_glareHardness: (preset?.glareHardness ?? 50) / 100,
          u_glareConvergence: (preset?.glareConvergence ?? 50) / 100,
          u_glareOppositeFactor: (preset?.glareOppositeFactor ?? 50) / 100,
          u_glareFactor: (preset?.glareFactor ?? 15) / 100,
          u_blurEdge: preset?.blurEdge ? 1 : 0,
          STEP: preset?.step ?? 9,
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
    fillShape,
    preset,
    shapeWidthRatio,
    shapeHeightRatio,
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
        className={cn(fallbackTransparent ? "bg-transparent" : "glass-panel rounded-lg", className)}
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
