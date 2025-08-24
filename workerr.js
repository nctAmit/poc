/* meshWorker.js
   - Parse GeoTIFF
   - Build a mesh on the GRID-CENTER lattice:
       centers = (width-1) x (height-1)
       triangles connect adjacent centers (2 per quad)
   - Vertex color by elevation (red→blue), optional hillshade baked
   - Compute normals
   - Returns typed arrays via Transferables
*/
self.importScripts('https://cdn.jsdelivr.net/npm/geotiff');

const NODATA_ELEV = -99999;

// ---------- utils ----------
const clamp01 = v => Math.max(0, Math.min(1, v));
const clamp255 = v => Math.max(0, Math.min(255, v | 0));
const degToRad = d => d * Math.PI / 180;

// red→blue ramp with optional gamma + invert
function elevToColor(h, minH, maxH, { colorInvert = false, gamma = 1.0 } = {}) {
  if (!isFinite(h) || maxH <= minH) return [200, 200, 200];
  let t = (h - minH) / (maxH - minH);
  t = clamp01(t);
  if (gamma && gamma !== 1) t = Math.pow(t, gamma);
  if (colorInvert) t = 1 - t;
  const r = 255 * (1 - t), g = 0, b = 255 * t;
  return [r | 0, g, b | 0];
}

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

function computeElevRange(elev, treatZeroAsNoData) {
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
  if (maxH === minH) maxH = minH + 1e-6;
  return { minH, maxH };
}

// Per-grid hillshade (used only to modulate vertex color if shadeStrength>0)
function computeGridHillshade(elev, width, height, minLon, maxLon, minLat, maxLat, zExaggeration, sunAzimuthDeg, sunAltitudeDeg) {
  const out = new Float32Array(width * height);
  const latMid = (minLat + maxLat) * 0.5;
  const mPerDegLat = 110540;
  const mPerDegLon = 111320 * Math.cos(degToRad(latMid));
  const dLon = (maxLon - minLon) / (width - 1 || 1);
  const dLat = (maxLat - minLat) / (height - 1 || 1);
  const dx = Math.abs(dLon) * mPerDegLon;
  const dy = Math.abs(dLat) * mPerDegLat;

  const az = degToRad(sunAzimuthDeg);   // 0=N, 90=E
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

// ------------- Build mesh on center lattice -------------
function buildCenterLatticeMesh({
  width, height,
  minLon, maxLon, minLat, maxLat,
  elev, zExaggeration,
  treatZeroAsNoData,
  colorParams, hsGrid, shadeStrength
}) {
  // centers grid size
  const cw = width - 1;
  const ch = height - 1;
  if (cw <= 0 || ch <= 0) {
    return {
      positions: new Float64Array(0),
      colors: new Uint8Array(0),
      indices: new Uint32Array(0)
    };
  }

  const lonStep = (maxLon - minLon) / (width - 1 || 1);
  const latStep = (maxLat - minLat) / (height - 1 || 1);
  const centerLonStep = lonStep; // center at mid of cell => same spacing
  const centerLatStep = latStep;

  const idx = (x, y) => y * width + x;
  const cidx = (x, y) => y * cw + x;

  // Prepare center elevation & validity
  const centerH = new Float64Array(cw * ch);
  const centerValid = new Uint8Array(cw * ch);

  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const tl = idx(x, y);
      const tr = idx(x + 1, y);
      const bl = idx(x, y + 1);
      const br = idx(x + 1, y + 1);

      const hTL = elev[tl], hTR = elev[tr], hBL = elev[bl], hBR = elev[br];

      // A center is valid only if all 4 surrounding samples are valid
      const valid =
        hTL !== NODATA_ELEV && hTR !== NODATA_ELEV &&
        hBL !== NODATA_ELEV && hBR !== NODATA_ELEV &&
        !(treatZeroAsNoData && (hTL === 0 || hTR === 0 || hBL === 0 || hBR === 0));

      if (valid) {
        const h = (hTL + hTR + hBL + hBR) * 0.25;
        centerH[cidx(x, y)] = h * zExaggeration;
        centerValid[cidx(x, y)] = 1;
      } else {
        centerH[cidx(x, y)] = 0;
        centerValid[cidx(x, y)] = 0;
      }
    }
  }

  // Build vertex arrays (one vertex per valid center)
  const positions = [];
  const colors = [];
  const map = new Int32Array(cw * ch);
  map.fill(-1);

  const shade = (r, g, b, hs) => {
    if (!shadeStrength) return [r, g, b];
    const s = clamp01(shadeStrength);
    const lit = hs * 255;
    const nr = clamp255((1 - s) * r + s * lit);
    const ng = clamp255((1 - s) * g + s * lit);
    const nb = clamp255((1 - s) * b + s * lit);
    return [nr, ng, nb];
  };

  let nextV = 0;
  for (let y = 0; y < ch; y++) {
    // center latitude = average of two row lats
    const latC = ( (maxLat - y * latStep) + (maxLat - (y + 1) * latStep) ) * 0.5;
    for (let x = 0; x < cw; x++) {
      // center longitude = average of two column lons
      const lonC = ( (minLon + x * lonStep) + (minLon + (x + 1) * lonStep) ) * 0.5;

      const i = cidx(x, y);
      if (!centerValid[i]) { map[i] = -1; continue; }

      const h = centerH[i];
      positions.push(latC, lonC, h);

      // Color by center elevation; note we pass the *pre-exaggeration* range scaled by zExaggeration to be consistent.
      // You can also feed already-scaled min/max from outside if preferred.
      let [r, g, b] = elevToColor(h, colorParams.minH * zExaggeration, colorParams.maxH * zExaggeration, colorParams);

      // Approx hillshade at center = average of the 4 surrounding sample hillshades (if available)
      if (hsGrid) {
        const tl = idx(x, y), tr = idx(x + 1, y), bl = idx(x, y + 1), br = idx(x + 1, y + 1);
        const hs = (hsGrid[tl] + hsGrid[tr] + hsGrid[bl] + hsGrid[br]) * 0.25;
        [r, g, b] = shade(r, g, b, hs);
      }

      colors.push(r, g, b, 255);
      map[i] = nextV++;
    }
  }

  // Triangulate the center lattice: for each 2x2 block of centers, 2 triangles
  const indices = [];
  for (let y = 0; y < ch - 1; y++) {
    for (let x = 0; x < cw - 1; x++) {
      const v00 = map[cidx(x, y)];
      const v10 = map[cidx(x + 1, y)];
      const v01 = map[cidx(x, y + 1)];
      const v11 = map[cidx(x + 1, y + 1)];

      if (v00 === -1 || v10 === -1 || v01 === -1 || v11 === -1) continue;

      // Choose a consistent diagonal; you can pick based on slope if you want
      indices.push(v00, v10, v11);
      indices.push(v00, v11, v01);
    }
  }

  return {
    positions: new Float64Array(positions),
    colors: new Uint8Array(colors),
    indices: new Uint32Array(indices),
  };
}

// ----------------- Worker entry -----------------
self.onmessage = async (e) => {
  const { id, type, payload } = e.data || {};
  try {
    if (type !== 'PROCESS_TIFF') throw new Error(`Unknown worker message type: ${type}`);

    const {
      arrayBuffer,
      zExaggeration = 1.0,
      colorInvert = false,
      gamma = 1.0,
      shadeStrength = 0.35,
      sunAzimuthDeg = 315,
      sunAltitudeDeg = 45,
      treatZeroAsNoData = true,
    } = payload || {};

    const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    const image = await tiff.getImage();
    const rasters = await image.readRasters(); // [0]=elev
    const width = image.getWidth();
    const height = image.getHeight();
    const bbox = image.getBoundingBox(); // [minX, minY, maxX, maxY]
    const [minLon, minLat, maxLon, maxLat] = bbox;

    const elev = rasters[0];

    // color range (pre-exaggeration; scaled later)
    const { minH, maxH } = computeElevRange(elev, treatZeroAsNoData);
    const colorParams = { minH, maxH, colorInvert, gamma };

    // optional hillshade (per source sample)
    const hsGrid = shadeStrength > 0
      ? computeGridHillshade(elev, width, height, minLon, maxLon, minLat, maxLat, zExaggeration, sunAzimuthDeg, sunAltitudeDeg)
      : null;

    // build mesh on center lattice
    const { positions, colors, indices } = buildCenterLatticeMesh({
      width, height, minLon, maxLon, minLat, maxLat,
      elev, zExaggeration, treatZeroAsNoData,
      colorParams, hsGrid, shadeStrength
    });

    const normals = computeNormals(positions, indices);

    self.postMessage({
      id,
      ok: true,
      result: {
        width, height, bbox,
        positions: positions.buffer,
        colors: colors.buffer,
        indices: indices.buffer,
        normals: normals.buffer
      }
    }, [positions.buffer, colors.buffer, indices.buffer, normals.buffer]);

  } catch (err) {
    self.postMessage({ id, ok: false, error: (err && err.message) || String(err) });
  }
};
