/**
 * Minimaler safetensors-Leser: 8 Byte Header-Länge (little endian), JSON-Header, Rohdaten.
 * Unterstützt F32, BF16 und F16 und liefert immer Float32Array.
 */

export interface TensorInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface SafeTensors {
  header: Record<string, TensorInfo>;
  get(name: string): { shape: number[]; data: Float32Array };
  names(): string[];
}

function f16ToF32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

export function parseSafeTensors(buf: ArrayBuffer): SafeTensors {
  const view = new DataView(buf);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(new Uint8Array(buf, 8, headerLen));
  const header = JSON.parse(headerJson) as Record<string, TensorInfo>;
  delete (header as Record<string, unknown>).__metadata__;
  const dataStart = 8 + headerLen;

  return {
    header,
    names: () => Object.keys(header),
    get(name) {
      const t = header[name];
      if (!t) throw new Error(`Tensor fehlt: ${name}`);
      const [a, b] = t.data_offsets;
      const n = t.shape.reduce((x, y) => x * y, 1);
      const out = new Float32Array(n);
      const start = dataStart + a;
      if (t.dtype === 'F32') {
        // Kopie (Offsets sind nicht immer 4-Byte-ausgerichtet)
        out.set(new Float32Array(buf.slice(start, dataStart + b)));
      } else if (t.dtype === 'BF16') {
        const u16 = new Uint16Array(buf.slice(start, dataStart + b));
        const u32 = new Uint32Array(out.buffer);
        for (let i = 0; i < n; i++) u32[i] = u16[i] << 16;
      } else if (t.dtype === 'F16') {
        const u16 = new Uint16Array(buf.slice(start, dataStart + b));
        for (let i = 0; i < n; i++) out[i] = f16ToF32(u16[i]);
      } else {
        throw new Error(`dtype ${t.dtype} wird nicht unterstützt (${name})`);
      }
      return { shape: t.shape, data: out };
    },
  };
}
