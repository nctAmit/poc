/* meshWorker.js
   - Parse GeoTIFF
   - Build center-lattice mesh
   - FLAT color per triangle (no interpolation): duplicate vertices per triangle, same RGBA
   - Optional uniform hillshade modulation per triangle center (shadeStrength>0)
   - Compute normals
   - Returns typed arrays via Transferables
*/
self.importScripts('https://cdn.jsdelivr.net/npm/geotiff');

const NODATA_ELEV = -99999;

// ---------- utils ----------
const clamp01 = v => Math.max(0, Math.min(1, v));
const clamp255 = v => Math.max(0, Math.min(255, v | 0));
const degToRad = d => d * Math.PI / 180;

function hillshadeFromNormal(nx, ny, nz, Lx, Ly, Lz) {
  return 0.25 + 0.75 * clamp01(nx * Lx + ny * Ly + nz * Lz);
}

// Simple central-diff hillshade at grid nodes
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

function computeNormals(positions, indices) {
  const normals = new Float32Array(positions.length); // zeros
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
  // normalize (per-vertex; with no sharing, this equals per-face)
  for (let i = 0; i < normals.length; i += 3) {
    const nx = normals[i], ny = normals[i + 1], nz = normals[i + 2];
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) { normals[i] = nx / len; normals[i + 1] = ny / len; normals[i + 2] = nz / len; }
  }
  return normals;
}

// ----------------- Flat-color center-lattice mesh -----------------
function buildFlatTriangleMeshFromCenters({
  width, height,
  minLon, maxLon, minLat, maxLat,
  elev, R, G, B, A,
  zExaggeration,
  treatZeroAsNoData,
  hsGrid, shadeStrength
}) {
  // centers grid size
  const cw = width - 1;
  const ch = height - 1;
  if (cw <= 0 || ch <= 0) {
    return {
      positions: new Float64Array(0),
      colors: new Uint8Array(0),
      indices: new Uint32Array(0),
    };
  }

  const lonStep = (maxLon - minLon) / (width - 1 || 1);
  const latStep = (maxLat - minLat) / (height - 1 || 1);

  const idx = (x, y) => y * width + x;
  const cidx = (x, y) => y * cw + x;

  // Precompute each center's height & color (avg of 4 corners), and validity
  const centerH = new Float64Array(cw * ch);
  const centerColor = new Uint8Array(cw * ch * 4);
  const centerValid = new Uint8Array(cw * ch);

  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const tl = idx(x, y), tr = idx(x + 1, y), bl = idx(x, y + 1), br = idx(x + 1, y + 1);
      const hTL = elev[tl], hTR = elev[tr], hBL = elev[bl], hBR = elev[br];

      const valid =
        hTL !== NODATA_ELEV && hTR !== NODATA_ELEV &&
        hBL !== NODATA_ELEV && hBR !== NODATA_ELEV &&
        !(treatZeroAsNoData && (hTL === 0 || hTR === 0 || hBL === 0 || hBR === 0));

      const ci = cidx(x, y);

      if (!valid) {
        centerValid[ci] = 0;
        centerH[ci] = 0;
        continue;
      }

      centerValid[ci] = 1;
      centerH[ci] = ((hTL + hTR + hBL + hBR) * 0.25) * zExaggeration;

      // Averaged center color from GeoTIFF RGB(A)
      const r = ((R[tl] + R[tr] + R[bl] + R[br]) * 0.25) | 0;
      const g = ((G[tl] + G[tr] + G[bl] + G[br]) * 0.25) | 0;
      const b = ((B[tl] + B[tr] + B[bl] + B[br]) * 0.25) | 0;
      const a = A ? ((A[tl] + A[tr] + A[bl] + A[br]) * 0.25) | 0 : 255;

      const o = ci * 4;
      centerColor[o] = r; centerColor[o + 1] = g; centerColor[o + 2] = b; centerColor[o + 3] = a;
    }
  }

  // Helper: center position (lat,lon,h)
  function centerPos(x, y) {
    const latC = ((maxLat - y * latStep) + (maxLat - (y + 1) * latStep)) * 0.5;
    const lonC = ((minLon + x * lonStep) + (minLon + (x + 1) * lonStep)) * 0.5;
    const h = centerH[cidx(x, y)];
    return [latC, lonC, h];
  }

  // Helper: center RGBA
  function centerRGBA(x, y) {
    const o = cidx(x, y) * 4;
    return [
      centerColor[o],
      centerColor[o + 1],
      centerColor[o + 2],
      centerColor[o + 3],
    ];
  }

  // Optional: uniform shading modulation per triangle center
  function modulateRGBA(r, g, b, x0, y0, x1, y1, x2, y2) {
    if (!shadeStrength || !hsGrid) return [r, g, b];
    const s = clamp01(shadeStrength);

    // Triangle center in grid indices space → approximate by averaging the three *corner sample* hillshades closest to centers
    // Pick nearest 4 corner samples around each involved center and average all 12 -> or simpler: average the 4 samples of the cell nearest to the tri centroid.
    const cx = (x0 + x1 + x2) / 3;
    const cy = (y0 + y1 + y2) / 3;

    // Map centroid (in center grid) back to nearest raster sample block around floor(cx), floor(cy)
    const gx = Math.max(0, Math.min(width - 2, Math.floor(cx)));
    const gy = Math.max(0, Math.min(height - 2, Math.floor(cy)));

    const tl = idx(gx, gy), tr = idx(gx + 1, gy), bl = idx(gx, gy + 1), br = idx(gx + 1, gy + 1);
    const hs = (hsGrid[tl] + hsGrid[tr] + hsGrid[bl] + hsGrid[br]) * 0.25; // 0..1

    const lit = hs * 255;
    const nr = clamp255((1 - s) * r + s * lit);
    const ng = clamp255((1 - s) * g + s * lit);
    const nb = clamp255((1 - s) * b + s * lit);
    return [nr, ng, nb];
  }

  // Count triangles first (valid 2x2 blocks of centers → 2 tris)
  let triCount = 0;
  for (let y = 0; y < ch - 1; y++) {
    for (let x = 0; x < cw - 1; x++) {
      const v00 = centerValid[cidx(x, y)];
      const v10 = centerValid[cidx(x + 1, y)];
      const v01 = centerValid[cidx(x, y + 1)];
      const v11 = centerValid[cidx(x + 1, y + 1)];
      if (v00 && v10 && v01 && v11) triCount += 2;
    }
  }

  // Allocate non-indexed-style buffers but keep indices for consistency (0..n-1)
  // We will duplicate vertices per triangle (flat color)
  const positions = new Float64Array(triCount * 3 * 3); // triCount * 3 verts * 3 comps
  const colors    = new Uint8Array (triCount * 3 * 4);  // triCount * 3 verts * RGBA
  const indices   = new Uint32Array(triCount * 3);      // 0..N-1

  let vPtr = 0; // vertex cursor (float triplets)
  let cPtr = 0; // color cursor (rgba quads)
  let iPtr = 0; // index cursor
  let vStart = 0;

  // Build triangles; for each, compute ONE color and assign to all 3 duplicated verts
  for (let y = 0; y < ch - 1; y++) {
    for (let x = 0; x < cw - 1; x++) {
      const validBlock =
        centerValid[cidx(x, y)] &&
        centerValid[cidx(x + 1, y)] &&
        centerValid[cidx(x, y + 1)] &&
        centerValid[cidx(x + 1, y + 1)];

      if (!validBlock) continue;

      // Tri 1: (v00, v10, v11)
      {
        const [lat0, lon0, h0] = centerPos(x, y);
        const [lat1, lon1, h1] = centerPos(x + 1, y);
        const [lat2, lon2, h2] = centerPos(x + 1, y + 1);

        // Average the three centers' colors → flat triangle color
        let [r0, g0, b0, a0] = centerRGBA(x, y);
        let [r1, g1, b1, a1] = centerRGBA(x + 1, y);
        let [r2, g2, b2, a2] = centerRGBA(x + 1, y + 1);

        let r = ((r0 + r1 + r2) / 3) | 0;
        let g = ((g0 + g1 + g2) / 3) | 0;
        let b = ((b0 + b1 + b2) / 3) | 0;
        let a = ((a0 + a1 + a2) / 3) | 0;

        // Optional uniform shading at triangle center
        [r, g, b] = modulateRGBA(r, g, b, x, y, x + 1, y, x + 1, y + 1);

        positions[vPtr++] = lat0; positions[vPtr++] = lon0; positions[vPtr++] = h0;
        positions[vPtr++] = lat1; positions[vPtr++] = lon1; positions[vPtr++] = h1;
        positions[vPtr++] = lat2; positions[vPtr++] = lon2; positions[vPtr++] = h2;

        colors[cPtr++] = r; colors[cPtr++] = g; colors[cPtr++] = b; colors[cPtr++] = a;
        colors[cPtr++] = r; colors[cPtr++] = g; colors[cPtr++] = b; colors[cPtr++] = a;
        colors[cPtr++] = r; colors[cPtr++] = g; colors[cPtr++] = b; colors[cPtr++] = a;

        indices[iPtr++] = vStart++;
        indices[iPtr++] = vStart++;
        indices[iPtr++] = vStart++;
      }

      // Tri 2: (v00, v11, v01)
      {
        const [lat0, lon0, h0] = centerPos(x, y);
        const [lat1, lon1, h1] = centerPos(x + 1, y + 1);
        const [lat2, lon2, h2] = centerPos(x, y + 1);

        let [r0, g0, b0, a0] = centerRGBA(x, y);
        let [r1, g1, b1, a1] = centerRGBA(x + 1, y + 1);
        let [r2, g2, b2, a2] = centerRGBA(x, y + 1);

        let r = ((r0 + r1 + r2) / 3) | 0;
        let g = ((g0 + g1 + g2) / 3) | 0;
        let b = ((b0 + b1 + b2) / 3) | 0;
        let a = ((a0 + a1 + a2) / 3) | 0;

        [r, g, b] = modulateRGBA(r, g, b, x, y, x + 1, y + 1, x, y + 1);

        positions[vPtr++] = lat0; positions[vPtr++] = lon0; positions[vPtr++] = h0;
        positions[vPtr++] = lat1; positions[vPtr++] = lon1; positions[vPtr++] = h1;
        positions[vPtr++] = lat2; positions[vPtr++] = lon2; positions[vPtr++] = h2;

        colors[cPtr++] = r; colors[cPtr++] = g; colors[cPtr++] = b; colors[cPtr++] = a;
        colors[cPtr++] = r; colors[cPtr++] = g; colors[cPtr++] = b; colors[cPtr++] = a;
        colors[cPtr++] = r; colors[cPtr++] = g; colors[cPtr++] = b; colors[cPtr++] = a;

        indices[iPtr++] = vStart++;
        indices[iPtr++] = vStart++;
        indices[iPtr++] = vStart++;
      }
    }
  }

  return { positions, colors, indices };
}

// ----------------- Worker entry -----------------
self.onmessage = async (e) => {
  const { id, type, payload } = e.data || {};
  try {
    if (type !== 'PROCESS_TIFF') throw new Error(`Unknown worker message type: ${type}`);

    const {
      arrayBuffer,
      zExaggeration = 1.0,
      // Flat color: set >0 only if you want *uniform* light/dark modulation per triangle
      shadeStrength = 0.0,
      sunAzimuthDeg = 315,
      sunAltitudeDeg = 45,
      treatZeroAsNoData = true,
    } = payload || {};

    const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    const image = await tiff.getImage();
    const rasters = await image.readRasters(); // [0]=elev, [1]=R, [2]=G, [3]=B, ([4]=A optional)
    const width  = image.getWidth();
    const height = image.getHeight();
    const bbox   = image.getBoundingBox(); // [minX, minY, maxX, maxY]
    const [minLon, minLat, maxLon, maxLat] = bbox;

    const elev = rasters[0];
    const R = rasters[1], G = rasters[2], B = rasters[3];
    const A = rasters.length > 4 ? rasters[4] : null;

    const hsGrid = shadeStrength > 0
      ? computeGridHillshade(elev, width, height, minLon, maxLon, minLat, maxLat, zExaggeration, sunAzimuthDeg, sunAltitudeDeg)
      : null;

    const { positions, colors, indices } = buildFlatTriangleMeshFromCenters({
      width, height, minLon, maxLon, minLat, maxLat,
      elev, R, G, B, A,
      zExaggeration,
      treatZeroAsNoData,
      hsGrid, shadeStrength
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
