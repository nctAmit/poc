/* meshWorker.js
   - Parse GeoTIFF
   - Build positions/colors/indices (solid quads → two triangles)
   - Color by elevation (red→blue), optional hillshade baked
   - Compute normals
   Returns typed arrays via Transferables
*/
self.importScripts('https://cdn.jsdelivr.net/npm/geotiff');

const NODATA_ELEV = -99999;

// ---------- utils ----------
const clamp01 = v => Math.max(0, Math.min(1, v));
const clamp255 = v => Math.max(0, Math.min(255, v | 0));
const degToRad = d => d * Math.PI / 180;

// Linear red→blue (low→high), with optional gamma + invert
function elevToColor(h, minH, maxH, { colorInvert = false, gamma = 1.0 } = {}) {
  if (!isFinite(h) || maxH <= minH) return [200, 200, 200]; // fallback gray
  let t = (h - minH) / (maxH - minH);      // 0..1
  t = clamp01(t);
  if (gamma && gamma !== 1) t = Math.pow(t, gamma);
  if (colorInvert) t = 1 - t;

  // Red→Blue
  const r = 255 * (1 - t);
  const g = 0;
  const b = 255 * t;
  return [r | 0, g, b | 0];
}

// Optional hillshade (to enhance depth cues even with same hues)
function hillshadeFromNormal(nx, ny, nz, Lx, Ly, Lz) {
  return 0.25 + 0.75 * clamp01(nx * Lx + ny * Ly + nz * Lz);
}

function computeNormals(positions, indices) {
  const normals = new Float32Array(positions.length);
  for (let i = 0; i < indices.length; i += 3) {
    const ia = indices[i] * 3, ib = indices[i + 1] * 3, ic = indices[i + 2] * 3;
    const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2];
    const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2];
    const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2];
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    normals[ia] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
    normals[ib] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
    normals[ic] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
  }
  for (let i = 0; i < normals.length; i += 3) {
    const nx = normals[i], ny = normals[i + 1], nz = normals[i + 2];
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) { normals[i] = nx / len; normals[i + 1] = ny / len; normals[i + 2] = nz / len; }
  }
  return normals;
}

// Compute min/max of valid elevations (ignores nodata and zeros if you treat zero as “no elev”)
function computeElevRange(elev, treatZeroAsNoData = true) {
  let minH = +Infinity, maxH = -Infinity, any = false;
  for (let i = 0; i < elev.length; i++) {
    const h = elev[i];
    if (h === NODATA_ELEV) continue;
    if (treatZeroAsNoData && h === 0) continue;
    if (!isFinite(h)) continue;
    if (h < minH) minH = h;
    if (h > maxH) maxH = h;
    any = true;
  }
  if (!any) { minH = 0; maxH = 1; }
  if (maxH === minH) maxH = minH + 1e-6; // avoid div/0
  return { minH, maxH };
}

// Simple per-grid hillshade from gradients (optional)
function computeGridHillshade(elev, width, height, minLon, maxLon, minLat, maxLat, zExaggeration, sunAzimuthDeg, sunAltitudeDeg) {
  const out = new Float32Array(width * height);

  const latMid = (minLat + maxLat) * 0.5;
  const mPerDegLat = 110540;
  const mPerDegLon = 111320 * Math.cos(degToRad(latMid));
  const dLon = (maxLon - minLon) / (width - 1 || 1);
  const dLat = (maxLat - minLat) / (height - 1 || 1);
  const dx = Math.abs(dLon) * mPerDegLon;
  const dy = Math.abs(dLat) * mPerDegLat;

  const az = degToRad(sunAzimuthDeg);   // 0=N,90=E
  const alt = degToRad(sunAltitudeDeg);
  const Lx = Math.sin(az) * Math.cos(alt);
  const Ly = Math.cos(az) * Math.cos(alt);
  const Lz = Math.sin(alt);
  const idx = (x, y) => y * width + x;

  for (let y = 0; y < height; y++) {
    const ym = Math.max(0, y - 1), yp = Math.min(height - 1, y + 1);
    for (let x = 0; x < width; x++) {
      const xm = Math.max(0, x - 1), xp = Math.min(width - 1, x + 1);
      const c = elev[idx(x, y)];
      if (c === NODATA_ELEV) { out[idx(x, y)] = 0.5; continue; }
      const em = elev[idx(xp, y)] !== NODATA_ELEV ? elev[idx(xp, y)] : c;
      const eM = elev[idx(xm, y)] !== NODATA_ELEV ? elev[idx(xm, y)] : c;
      const en = elev[idx(x, yp)] !== NODATA_ELEV ? elev[idx(x, yp)] : c;
      const eN = elev[idx(x, ym)] !== NODATA_ELEV ? elev[idx(x, ym)] : c;
      const dzdx = ((em - eM) * 0.5 * zExaggeration) / (dx || 1);
      const dzdy = ((en - eN) * 0.5 * zExaggeration) / (dy || 1);
      let nx = -dzdx, ny = -dzdy, nz = 1;
      const inv = 1 / Math.hypot(nx, ny, nz);
      nx *= inv; ny *= inv; nz *= inv;
      out[idx(x, y)] = hillshadeFromNormal(nx, ny, nz, Lx, Ly, Lz); // 0..1
    }
  }
  return out;
}

// Build solid-quad mesh; pick color for each cell from TL vertex elevation
function buildSolidRectGrid(positionsAll, indexMap, width, height, elevGrid, colorParams, hillshadeGrid, shadeStrength) {
  // Count quads with all four vertices present
  let quadCount = 0;
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x;
      const tr = tl + 1;
      const bl = (y + 1) * width + x;
      const br = bl + 1;
      if (indexMap[tl] !== -1 && indexMap[tr] !== -1 && indexMap[bl] !== -1 && indexMap[br] !== -1) quadCount++;
    }
  }

  const outPositions = new Float64Array(quadCount * 4 * 3);
  const outColors    = new Uint8Array(quadCount * 4 * 4);
  const outIndices   = new Uint32Array(quadCount * 6);

  let vtx = 0;
  let idx = 0;

  const shade = (r, g, b, hs) => {
    if (!shadeStrength) return [r, g, b];
    const s = clamp01(shadeStrength);
    const lit = hs * 255;
    const nr = clamp255((1 - s) * r + s * lit);
    const ng = clamp255((1 - s) * g + s * lit);
    const nb = clamp255((1 - s) * b + s * lit);
    return [nr, ng, nb];
  };

  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x;
      const tr = tl + 1;
      const bl = (y + 1) * width + x;
      const br = bl + 1;

      const i0 = indexMap[tl], i1 = indexMap[bl], i2 = indexMap[br], i3 = indexMap[tr];
      if (i0 === -1 || i1 === -1 || i2 === -1 || i3 === -1) continue;

      // Base color from elevation at TL
      const hTL = elevGrid[tl];
      let [r, g, b] = elevToColor(hTL, colorParams.minH, colorParams.maxH, colorParams);

      // Optional hillshade baked
      if (hillshadeGrid) {
        const hs = hillshadeGrid[tl];
        [r, g, b] = shade(r, g, b, hs);
      }

      const addCopy = (srcIndex) => {
        const pBase = srcIndex * 3, o = vtx * 3;
        outPositions[o]     = positionsAll[pBase];
        outPositions[o + 1] = positionsAll[pBase + 1];
        outPositions[o + 2] = positionsAll[pBase + 2];

        const co = vtx * 4;
        outColors[co]     = r;
        outColors[co + 1] = g;
        outColors[co + 2] = b;
        outColors[co + 3] = 255;
        vtx++;
      };

      addCopy(i0); addCopy(i1); addCopy(i2); addCopy(i3);

      const base = vtx - 4;
      outIndices[idx++] = base;     outIndices[idx++] = base + 1; outIndices[idx++] = base + 2;
      outIndices[idx++] = base;     outIndices[idx++] = base + 2; outIndices[idx++] = base + 3;
    }
  }

  return { outPositions, outColors, outIndices };
}

self.onmessage = async (e) => {
  const { id, type, payload } = e.data || {};
  try {
    if (type !== 'PROCESS_TIFF') throw new Error(`Unknown worker message type: ${type}`);

    const {
      arrayBuffer,
      deltaZ = 0,
      zExaggeration = 1.0,     // makes small relief visible
      // elevation color ramp controls
      colorInvert = false,
      gamma = 1.0,
      // hillshade (set to 0 to disable baking)
      shadeStrength = 0.5,
      sunAzimuthDeg = 315,
      sunAltitudeDeg = 45,
      // treat zero as "no elevation" like before
      treatZeroAsNoData = true,
    } = payload || {};

    const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    const image = await tiff.getImage();
    const rasters = await image.readRasters(); // [0]=elev, others ignored
    const width = image.getWidth();
    const height = image.getHeight();

    const bbox = image.getBoundingBox(); // [minX, minY, maxX, maxY]
    const [minLon, minLat, maxLon, maxLat] = bbox;
    const lonStep = (maxLon - minLon) / (width - 1 || 1);
    const latStep = (maxLat - minLat) / (height - 1 || 1);

    const elev = rasters[0];

    // Elevation range for coloring
    const { minH, maxH } = computeElevRange(elev, treatZeroAsNoData);
    const colorParams = { minH, maxH, colorInvert, gamma };

    // Optional hillshade grid (in source grid space)
    const hsGrid = shadeStrength > 0
      ? computeGridHillshade(elev, width, height, minLon, maxLon, minLat, maxLat, zExaggeration, sunAzimuthDeg, sunAltitudeDeg)
      : null;

    // Build vertices (skip nodata / zero if configured)
    const positions = [];
    const indexMap = new Int32Array(width * height);
    indexMap.fill(-1);

    let nextIdx = 0;
    for (let y = 0; y < height; y++) {
      const lat = maxLat - y * latStep;
      for (let x = 0; x < width; x++) {
        const lon = minLon + x * lonStep;
        const i = y * width + x;

        let h = elev[i];
        if (h === NODATA_ELEV || (!h && treatZeroAsNoData)) {
          indexMap[i] = -1;
          continue;
        }
        h = (h + Number(deltaZ || 0)) * Number(zExaggeration || 1);

        positions.push(lat, lon, h);
        indexMap[i] = nextIdx++;
      }
    }

    const posArr = new Float64Array(positions);

    // Build quads with elevation-based coloring
    const { outPositions, outColors, outIndices } =
      buildSolidRectGrid(posArr, indexMap, width, height, elev, colorParams, hsGrid, shadeStrength);

    const normals = computeNormals(outPositions, outIndices);

    self.postMessage({
      id,
      ok: true,
      result: {
        width,
        height,
        bbox,
        positions: outPositions.buffer,
        colors: outColors.buffer,
        indices: outIndices.buffer,
        normals: normals.buffer
      }
    }, [outPositions.buffer, outColors.buffer, outIndices.buffer, normals.buffer]);

  } catch (err) {
    self.postMessage({ id, ok: false, error: (err && err.message) || String(err) });
  }
};
