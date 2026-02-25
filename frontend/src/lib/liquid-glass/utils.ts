export function computeGaussianKernelByRadius(radius: number): number[] {
  const sigma = radius / 3.0;
  const kernel: number[] = [];
  let sum = 0;
  for (let i = 0; i <= radius; i++) {
    const weight = Math.exp(-0.5 * (i * i) / (sigma * sigma));
    kernel.push(weight);
    sum += i === 0 ? weight : weight * 2;
  }
  return kernel.map((w) => w / sum);
}
