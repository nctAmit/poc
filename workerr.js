/* meshWorker.js
   - Parse GeoTIFF
   - Build a mesh on the GRID-CENTER lattice:
       centers = (width-1) x (height-1)
       triangles connect adjacent centers (2 per 4-center quad)
   - Vertex color from GeoTIFF RGB(A) averaged from the 4 surrounding samples
   - Optional hillshade baked into those colors
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

// ------------- Build mesh on center lattice using GeoTIFF colors -------------
function buildCenterLatticeMesh({
  width, height,
  minLon, maxLon, minLat, maxLat,
  elev, R, G, B, A,
  zExaggeration,
  treatZeroAsNoData,
  hsGrid, shadeStrength
}) {
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

  const idx = (x, y) => y * width + x;
  const cidx = (x, y) => y * cw + x;

  // Center attributes
  const centerH = new Float64Array(cw * ch);
  const centerValid = new Uint8Array(cw * ch);
  const centerColor = new Uint8Array(cw * ch * 4);

  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const tl = idx(x, y);
      const tr = idx(x + 1, y);
      const bl = idx(x, y + 1);
      const br = idx(x + 1, y + 1);

      const hTL = elev[tl], hTR = elev[tr], hBL = elev[bl], hBR = elev[br];

      const valid =
        hTL !== NODATA_ELEV && hTR !== NODATA_ELEV &&
        hBL !== NODATA_ELEV && hBR !== NODATA_ELEV &&
        !(treatZeroAsNoData && (hTL === 0 || hTR === 0 || hBL === 0 || hBR === 0));

      const iC = cidx(x, y);

      if (!valid) {
        centerValid[iC] = 0;
        centerH[iC] = 0;
        // color left uninitialized; vertex will be skipped anyway
        continue;
      }

      const h = (hTL + hTR + hBL + hBR) * 0.25 * zExaggeration;
      centerH[iC] = h;
      centerValid[iC] = 1;

      // Average GeoTIFF color (use 4-band RGBA if available, else RGB with A=255)
      const r = ((R[tl] + R[tr] + R[bl] + R[br]) * 0.25) | 0;
      const g = ((G[tl] + G[tr] + G[bl] + G[br]) * 0.25) | 0;
      const b = ((B[tl] + B[tr] + B[bl] + B[br]) * 0.25) | 0;
      const a = A ? ((A[tl] + A[tr] + A[bl] + A[br]) * 0.25) | 0 : 255;

      const o = iC * 4;
      centerColor[o] = r;
      centerColor[o + 1] = g;
      centerColor[o + 2] = b;
      centerColor[o + 3] = a;
    }
  }

  const positions = [];
  const colors = [];
  const map = new Int32Array(cw * ch);
  map.fill(-1);

  const modulateWithShade = (r, g, b, hs) => {
    if (!shadeStrength) return [r, g, b];
    const s = clamp01(shadeStrength);
    const lit = hs * 255;
    return [
      clamp255((1 - s) * r + s * lit),
      clamp255((1 - s) * g + s * lit),
      clamp255((1 - s) * b + s * lit)
    ];
  };

  let nextV = 0;
  for (let y = 0; y < ch; y++) {
    const latC = ( (maxLat - y * latStep) + (maxLat - (y + 1) * latStep) ) * 0.5;
    for (let x = 0; x < cw; x++) {
      const lonC = ( (minLon + x * lonStep) + (minLon + (x + 1) * lonStep) ) * 0.5;
      const iC = cidx(x, y);
      if (!centerValid[iC]) { map[iC] = -1; continue; }

      const h = centerH[iC];
      positions.push(latC, lonC, h);

      let r = centerColor[iC*4], g = centerColor[iC*4+1], b = centerColor[iC*4+2], a = centerColor[iC*4+3];

      // Approx hillshade at center = average of the 4 surrounding sample hillshades (if available)
      if (hsGrid) {
        const tl = idx(x, y), tr = idx(x + 1, y), bl = idx(x, y + 1), br = idx(x + 1, y + 1);
        const hs = (hsGrid[tl] + hsGrid[tr] + hsGrid[bl] + hsGrid[br]) * 0.25;
        [r, g, b] = modulateWithShade(r, g, b, hs);
      }

      colors.push(r, g, b, a);
      map[iC] = nextV++;
    }
  }

  // Triangles on center lattice: 2 per 2x2 block of centers
  const indices = [];
  const cidx2 = (x, y) => y * cw + x;

  for (let y = 0; y < ch - 1; y++) {
    for (let x = 0; x < cw - 1; x++) {
      const v00 = map[cidx2(x, y)];
      const v10 = map[cidx2(x + 1, y)];
      const v01 = map[cidx2(x, y + 1)];
      const v11 = map[cidx2(x + 1, y + 1)];
      if (v00 === -1 || v10 === -1 || v01 === -1 || v11 === -1) continue;

      // consistent diagonal
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
      // baked shading controls (set shadeStrength=0 to disable)
      shadeStrength = 0.0,
      sunAzimuthDeg = 315,
      sunAltitudeDeg = 45,
      // treat zero elevation as NODATA (matches many raster exports)
      treatZeroAsNoData = true,
    } = payload || {};

    const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    const image = await tiff.getImage();
    const rasters = await image.readRasters(); // Expect [0]=elev, [1]=R, [2]=G, [3]=B, ([4]=A optional)
    const width = image.getWidth();
    const height = image.getHeight();
    const bbox = image.getBoundingBox(); // [minX, minY, maxX, maxY]
    const [minLon, minLat, maxLon, maxLat] = bbox;

    const elev = rasters[0];
    const R = rasters[1], G = rasters[2], B = rasters[3];
    const A = rasters.length > 4 ? rasters[4] : null;

    // optional hillshade per source sample
    const hsGrid = shadeStrength > 0
      ? computeGridHillshade(elev, width, height, minLon, maxLon, minLat, maxLat, zExaggeration, sunAzimuthDeg, sunAltitudeDeg)
      : null;

    // build mesh on center lattice using GeoTIFF colors
    const { positions, colors, indices } = buildCenterLatticeMesh({
      width, height, minLon, maxLon, minLat, maxLat,
      elev, R, G, B, A,
      zExaggeration, treatZeroAsNoData,
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
