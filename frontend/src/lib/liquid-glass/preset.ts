/**
 * Liquid Glass preset from liquid-glass-studio export.
 * Used for the Predict input bar so the glass matches the intended look.
 * Source: https://github.com/iyinchao/liquid-glass-studio
 */
export interface LiquidGlassPresetControls {
  refThickness?: number;
  refFactor?: number;
  refDispersion?: number;
  refFresnelRange?: number;
  refFresnelHardness?: number;
  refFresnelFactor?: number;
  glareRange?: number;
  glareHardness?: number;
  glareFactor?: number;
  glareConvergence?: number;
  glareOppositeFactor?: number;
  glareAngle?: number;
  blurRadius?: number;
  blurEdge?: boolean;
  tint?: { r: number; g: number; b: number; a: number };
  shadowExpand?: number;
  shadowFactor?: number;
  shadowPosition?: { x: number; y: number };
  shapeWidth?: number;
  shapeHeight?: number;
  shapeRadius?: number;
  shapeRoundness?: number;
  mergeRate?: number;
  showShape1?: boolean;
  step?: number;
}

/** Preset from liquid-glass-studio export 2026-02-25T23:58:41 — exact values from repo, no custom aesthetics. */
export const LIQUID_GLASS_BAR_PRESET: LiquidGlassPresetControls = {
  refThickness: 20,
  refFactor: 1.4,
  refDispersion: 7,
  refFresnelRange: 30,
  refFresnelHardness: 20,
  refFresnelFactor: 20,
  glareRange: 30,
  glareHardness: 20,
  glareFactor: 90,
  glareConvergence: 50,
  glareOppositeFactor: 80,
  glareAngle: -45,
  blurRadius: 1,
  blurEdge: true,
  tint: { r: 255, g: 255, b: 255, a: 0 },
  shadowExpand: 25,
  shadowFactor: 15,
  shadowPosition: { x: 0, y: -10 },
  shapeWidth: 640,
  shapeHeight: 72,
  shapeRadius: 80,
  shapeRoundness: 5,
  mergeRate: 0.08,
  showShape1: false,
  step: 9,
};
