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

// ---------- Center-lattice (no overlaps at tile seams) ----------
function buildFlatTriangleMeshFromCenters_NoOverlap({
  width, height,
  minLon, maxLon, minLat, maxLat,
  elev, R, G, B, A,
  zExaggeration,
  treatZeroAsNoData,
  hsGrid, shadeStrength,
  halfOpenEdges = true // skip last col/row of center-quads to avoid duplicates w/ neighbors
}) {
  const cw = width - 1, ch = height - 1; // centers = cw*ch
  if (cw <= 1 || ch <= 1) {
    return { positions: new Float64Array(0), colors: new Uint8Array(0), indices: new Uint32Array(0) };
  }

  const lonStep = (maxLon - minLon) / (width - 1 || 1);
  const latStep = (maxLat - minLat) / (height - 1 || 1);
  const idx  = (x, y) => y * width + x;
  const cidx = (x, y) => y * cw + x;

  // Precompute valid centers (need 4 corner samples)
  const cH = new Float64Array(cw * ch);
  const cRGBA = new Uint8Array(cw * ch * 4);
  const cValid = new Uint8Array(cw * ch);

  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const tl = idx(x, y), tr = idx(x + 1, y), bl = idx(x, y + 1), br = idx(x + 1, y + 1);
      const hTL = elev[tl], hTR = elev[tr], hBL = elev[bl], hBR = elev[br];

      const valid =
        hTL !== NODATA_ELEV && hTR !== NODATA_ELEV &&
        hBL !== NODATA_ELEV && hBR !== NODATA_ELEV &&
        !(treatZeroAsNoData && (hTL === 0 || hTR === 0 || hBL === 0 || hBR === 0));

      const ci = cidx(x, y);
      cValid[ci] = valid ? 1 : 0;

      if (valid) {
        cH[ci] = ((hTL + hTR + hBL + hBR) * 0.25) * zExaggeration;

        const r = ((R[tl] + R[tr] + R[bl] + R[br]) * 0.25) | 0;
        const g = ((G[tl] + G[tr] + G[bl] + G[br]) * 0.25) | 0;
        const b = ((B[tl] + B[tr] + B[bl] + B[br]) * 0.25) | 0;
        const a = A ? ((A[tl] + A[tr] + A[bl] + A[br]) * 0.25) | 0 : 255;

        const o = ci * 4;
        cRGBA[o] = r; cRGBA[o + 1] = g; cRGBA[o + 2] = b; cRGBA[o + 3] = a;
      }
    }
  }

  // helpers for center pos/color
  function centerPos(x, y) {
    const latC = ((maxLat - y * latStep) + (maxLat - (y + 1) * latStep)) * 0.5;
    const lonC = ((minLon + x * lonStep) + (minLon + (x + 1) * lonStep)) * 0.5;
    const h = cH[cidx(x, y)];
    return [latC, lonC, h];
  }
  function centerRGBA(x, y) {
    const o = cidx(x, y) * 4;
    return [cRGBA[o], cRGBA[o + 1], cRGBA[o + 2], cRGBA[o + 3]];
  }

  // optional uniform shading modulation per-triangle (use centroid in center-grid)
  function modulateRGB(r, g, b, cx, cy) {
    if (!shadeStrength || !hsGrid) return [r, g, b];
    const s = Math.max(0, Math.min(1, shadeStrength));

    // map centroid to nearest raster 2×2 block
    const gx = Math.max(0, Math.min(width  - 2, Math.floor(cx)));
    const gy = Math.max(0, Math.min(height - 2, Math.floor(cy)));
    const tl = idx(gx, gy), tr = idx(gx + 1, gy), bl = idx(gx, gy + 1), br = idx(gx + 1, gy + 1);
    const hs = (hsGrid[tl] + hsGrid[tr] + hsGrid[bl] + hsGrid[br]) * 0.25; // 0..1
    const lit = (hs * 255) | 0;
    return [clamp255((1 - s) * r + s * lit), clamp255((1 - s) * g + s * lit), clamp255((1 - s) * b + s * lit)];
  }

  // Decide iteration bounds for 2×2 blocks of centers
  const maxX = halfOpenEdges ? (cw - 1) : (cw - 0);
  const maxY = halfOpenEdges ? (ch - 1) : (ch - 0);

  // Count triangles (2 per valid 2×2 block of centers)
  let triCount = 0;
  for (let y = 0; y < maxY - 1; y++) {
    for (let x = 0; x < maxX - 1; x++) {
      if (cValid[cidx(x, y)] && cValid[cidx(x + 1, y)] && cValid[cidx(x, y + 1)] && cValid[cidx(x + 1, y + 1)]) {
        triCount += 2;
      }
    }
  }

  const positions = new Float64Array(triCount * 3 * 3);
  const colors    = new Uint8Array (triCount * 3 * 4);
  const indices   = new Uint32Array(triCount * 3);

  let vPtr = 0, cPtr = 0, iPtr = 0, vStart = 0;

  // Build triangles across 2×2 center blocks (consistent diagonal)
  for (let y = 0; y < maxY - 1; y++) {
    for (let x = 0; x < maxX - 1; x++) {
      const v00 = cValid[cidx(x, y)];
      const v10 = cValid[cidx(x + 1, y)];
      const v01 = cValid[cidx(x, y + 1)];
      const v11 = cValid[cidx(x + 1, y + 1)];
      if (!(v00 && v10 && v01 && v11)) continue;

      const p00 = centerPos(x, y);
      const p10 = centerPos(x + 1, y);
      const p01 = centerPos(x, y + 1);
      const p11 = centerPos(x + 1, y + 1);

      let c00 = centerRGBA(x, y);
      let c10 = centerRGBA(x + 1, y);
      let c01 = centerRGBA(x, y + 1);
      let c11 = centerRGBA(x + 1, y + 1);

      // Triangle 1: (00, 10, 11)
      {
        let r = ((c00[0] + c10[0] + c11[0]) / 3) | 0;
        let g = ((c00[1] + c10[1] + c11[1]) / 3) | 0;
        let b = ((c00[2] + c10[2] + c11[2]) / 3) | 0;
        let a = ((c00[3] + c10[3] + c11[3]) / 3) | 0;

        // centroid in center-grid coords
        [r, g, b] = modulateRGB(r, g, b, x + (2/3), y + (1/3)); // rough centroid of (00,10,11)

        positions[vPtr++] = p00[0]; positions[vPtr++] = p00[1]; positions[vPtr++] = p00[2];
        positions[vPtr++] = p10[0]; positions[vPtr++] = p10[1]; positions[vPtr++] = p10[2];
        positions[vPtr++] = p11[0]; positions[vPtr++] = p11[1]; positions[vPtr++] = p11[2];

        colors[cPtr++] = r; colors[cPtr++] = g; colors[cPtr++] = b; colors[cPtr++] = a;
        colors[cPtr++] = r; colors[cPtr++] = g; colors[cPtr++] = b; colors[cPtr++] = a;
        colors[cPtr++] = r; colors[cPtr++] = g; colors[cPtr++] = b; colors[cPtr++] = a;

        indices[iPtr++] = vStart++;
        indices[iPtr++] = vStart++;
        indices[iPtr++] = vStart++;
      }

      // Triangle 2: (00, 11, 01)
      {
        let r = ((c00[0] + c11[0] + c01[0]) / 3) | 0;
        let g = ((c00[1] + c11[1] + c01[1]) / 3) | 0;
        let b = ((c00[2] + c11[2] + c01[2]) / 3) | 0;
        let a = ((c00[3] + c11[3] + c01[3]) / 3) | 0;

        [r, g, b] = modulateRGB(r, g, b, x + (1/3), y + (2/3)); // rough centroid of (00,11,01)

        positions[vPtr++] = p00[0]; positions[vPtr++] = p00[1]; positions[vPtr++] = p00[2];
        positions[vPtr++] = p11[0]; positions[vPtr++] = p11[1]; positions[vPtr++] = p11[2];
        positions[vPtr++] = p01[0]; positions[vPtr++] = p01[1]; positions[vPtr++] = p01[2];

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

