/* meshWorker.js */
/* CPU-bound pipeline:
   - Parse GeoTIFF
   - Build shared grid (one vertex per raster sample)
   - Compute smooth normals on shared grid
   - Emit center-fan per cell (4 tris) with flat per-cell color (from TL) and shared normals
   Returns typed arrays via Transferables
*/

self.importScripts('https://cdn.jsdelivr.net/npm/geotiff');

const NODATA_ELEV = -99999;

// ---------------- utils ----------------
function degToRad(d) { return d * Math.PI / 180; }

// Simple heuristic: if bbox numbers are way beyond degrees, assume Web Mercator (EPSG:3857)
function bboxIsWebMercator(bbox) {
  const [minX, minY, maxX, maxY] = bbox;
  const maxAbs = Math.max(Math.abs(minX), Math.abs(minY), Math.abs(maxX), Math.abs(maxY));
  return maxAbs > 360; // > degrees range → likely meters
}

function webMercatorToLonLat(x, y) {
  const R = 6378137.0;
  const lon = (x / R) * (180 / Math.PI);
  const lat = (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) * (180 / Math.PI);
  return [lon, lat];
}

// Per-vertex normals accumulated from triangle faces (positions in any consistent space)
function computeNormals(positions /* Float64Array */, indices /* Uint32Array */) {
  const normals = new Float32Array(positions.length); // zeroed
  for (let i = 0; i < indices.length; i += 3) {
    const ia = indices[i] * 3, ib = indices[i + 1] * 3, ic = indices[i + 2] * 3;

    const ax = positions[ia],     ay = positions[ia + 1],     az = positions[ia + 2];
    const bx = positions[ib],     by = positions[ib + 1],     bz = positions[ib + 2];
    const cx = positions[ic],     cy = positions[ic + 1],     cz = positions[ic + 2];

    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;

    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;

    normals[ia]     += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
    normals[ib]     += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
    normals[ic]     += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
  }
  // normalize
  for (let i = 0; i < normals.length; i += 3) {
    const nx = normals[i], ny = normals[i + 1], nz = normals[i + 2];
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) {
      normals[i]     = nx / len;
      normals[i + 1] = ny / len;
      normals[i + 2] = nz / len;
    }
  }
  return normals;
}

function buildSharedGridIndices(indexMap, width, height) {
  const arr = [];
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x;
      const tr = tl + 1;
      const bl = (y + 1) * width + x;
      const br = bl + 1;
      const iTL = indexMap[tl], iTR = indexMap[tr], iBL = indexMap[bl], iBR = indexMap[br];
      if (iTL === -1 || iTR === -1 || iBL === -1 || iBR === -1) continue;
      // shared-vertex grid for normal calc (two tris per cell)
      arr.push(iTL, iTR, iBR,  iTL, iBR, iBL);
    }
  }
  return new Uint32Array(arr);
}

function normalize3(x, y, z) {
  const L = Math.hypot(x, y, z);
  return L > 0 ? [x / L, y / L, z / L] : [0, 0, 1];
}

// Build center-fan per cell with flat color while copying smooth normals from shared grid
function buildCenterFanWithSmoothNormals(positionsAll, colorsAll, sharedNormals, indexMap, width, height) {
  // Count valid quads
  let quadCount = 0;
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x, tr = tl + 1, bl = (y + 1) * width + x, br = bl + 1;
      if (indexMap[tl] !== -1 && indexMap[tr] !== -1 && indexMap[bl] !== -1 && indexMap[br] !== -1) quadCount++;
    }
  }

  const VERTS_PER_QUAD = 5;   // TL, BL, BR, TR, C
  const TRIS_PER_QUAD  = 4;   // fan around center
  const outPositions = new Float64Array(quadCount * VERTS_PER_QUAD * 3);
  const outColors    = new Uint8Array (quadCount * VERTS_PER_QUAD * 4);
  const outNormals   = new Float32Array(quadCount * VERTS_PER_QUAD * 3);
  const outIndices   = new Uint32Array(quadCount * TRIS_PER_QUAD  * 3);

  let vtx = 0, idx = 0;

  function pushDup(srcIndex, r, g, b, a, nx, ny, nz) {
    const pBase = srcIndex * 3, v3 = vtx * 3, c4 = vtx * 4;
    outPositions[v3]     = positionsAll[pBase];
    outPositions[v3 + 1] = positionsAll[pBase + 1];
    outPositions[v3 + 2] = positionsAll[pBase + 2];

    outNormals[v3]     = nx;
    outNormals[v3 + 1] = ny;
    outNormals[v3 + 2] = nz;

    outColors[c4]     = r;
    outColors[c4 + 1] = g;
    outColors[c4 + 2] = b;
    outColors[c4 + 3] = a;

    return vtx++;
  }

  function normAt(i) {
    const b = i * 3;
    return [sharedNormals[b], sharedNormals[b + 1], sharedNormals[b + 2]];
  }

  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x, tr = tl + 1, bl = (y + 1) * width + x, br = bl + 1;
      const iTL = indexMap[tl], iTR = indexMap[tr], iBL = indexMap[bl], iBR = indexMap[br];
      if (iTL === -1 || iTR === -1 || iBL === -1 || iBR === -1) continue;

      // Per-cell flat color from TL
      const cBase = iTL * 4;
      const r = colorsAll[cBase]     ?? 200;
      const g = colorsAll[cBase + 1] ?? 200;
      const b = colorsAll[cBase + 2] ?? 200;
      const a = colorsAll[cBase + 3] ?? 255;

      // Corner normals from shared grid (smooth across cells)
      const nTL = normAt(iTL), nTR = normAt(iTR), nBL = normAt(iBL), nBR = normAt(iBR);

      // Corner duplicates (positions duplicated, normals copied from shared)
      const vTL = pushDup(iTL, r, g, b, a, nTL[0], nTL[1], nTL[2]);
      const vBL = pushDup(iBL, r, g, b, a, nBL[0], nBL[1], nBL[2]);
      const vBR = pushDup(iBR, r, g, b, a, nBR[0], nBR[1], nBR[2]);
      const vTR = pushDup(iTR, r, g, b, a, nTR[0], nTR[1], nTR[2]);

      // Center vertex = avg of 4 corners (pos + normal)
      const tlP = iTL * 3, blP = iBL * 3, brP = iBR * 3, trP = iTR * 3;
      const cLat = (positionsAll[tlP] + positionsAll[blP] + positionsAll[brP] + positionsAll[trP]) * 0.25;
      const cLon = (positionsAll[tlP + 1] + positionsAll[blP + 1] + positionsAll[brP + 1] + positionsAll[trP + 1]) * 0.25;
      const cH   = (positionsAll[tlP + 2] + positionsAll[blP + 2] + positionsAll[brP + 2] + positionsAll[trP + 2]) * 0.25;

      const nxC = (nTL[0] + nTR[0] + nBL[0] + nBR[0]) * 0.25;
      const nyC = (nTL[1] + nTR[1] + nBL[1] + nBR[1]) * 0.25;
      const nzC = (nTL[2] + nTR[2] + nBL[2] + nBR[2]) * 0.25;
      const cNorm = normalize3(nxC, nyC, nzC);

      const vC = vtx;
      const v3 = vC * 3, c4 = vC * 4;
      outPositions[v3]     = cLat;
      outPositions[v3 + 1] = cLon;
      outPositions[v3 + 2] = cH;
      outNormals[v3]       = cNorm[0];
      outNormals[v3 + 1]   = cNorm[1];
      outNormals[v3 + 2]   = cNorm[2];
      outColors[c4]        = r;
      outColors[c4 + 1]    = g;
      outColors[c4 + 2]    = b;
      outColors[c4 + 3]    = a;
      vtx++;

      // Fan (counter-clockwise): (C, TL, BL), (C, BL, BR), (C, BR, TR), (C, TR, TL)
      outIndices[idx++] = vC;  outIndices[idx++] = vTL; outIndices[idx++] = vBL;
      outIndices[idx++] = vC;  outIndices[idx++] = vBL; outIndices[idx++] = vBR;
      outIndices[idx++] = vC;  outIndices[idx++] = vBR; outIndices[idx++] = vTR;
      outIndices[idx++] = vC;  outIndices[idx++] = vTR; outIndices[idx++] = vTL;
    }
  }

  return { outPositions, outColors, outIndices, outNormals };
}

// --------------- worker ---------------
self.onmessage = async (e) => {
  const { id, type, payload } = e.data || {};
  try {
    if (type !== 'PROCESS_TIFF') throw new Error(`Unknown worker message type: ${type}`);

    const { arrayBuffer, deltaZ = 0 } = payload;

    const tiff    = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    const image   = await tiff.getImage();
    const rasters = await image.readRasters(); // expect [0]=elev, [1]=R, [2]=G, [3]=B when available
    const width   = image.getWidth();
    const height  = image.getHeight();

    const bboxSrc = image.getBoundingBox(); // [minX, minY, maxX, maxY] in source CRS
    const isWM = bboxIsWebMercator(bboxSrc);

    const minX = bboxSrc[0], minY = bboxSrc[1], maxX = bboxSrc[2], maxY = bboxSrc[3];
    const stepX = (maxX - minX) / (width  - 1 || 1);
    const stepY = (maxY - minY) / (height - 1 || 1);

    const elev = rasters[0];
    const hasRGB = rasters.length >= 4;
    const R = hasRGB ? rasters[1] : null;
    const G = hasRGB ? rasters[2] : null;
    const B = hasRGB ? rasters[3] : null;

    // Shared grid data
    const positions = []; // [lat, lon, h] (lon/lat in degrees)
    const colors    = []; // RGBA
    const indexMap  = new Int32Array(width * height);
    indexMap.fill(-1);

    let nextIdx = 0;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;

        // Sample source coords
        const srcX = minX + x * stepX;
        const srcY = maxY - y * stepY;

        // Convert to lon/lat if bbox is Web Mercator, else assume degrees already
        let lon, lat;
        if (isWM) {
          const ll = webMercatorToLonLat(srcX, srcY);
          lon = ll[0]; lat = ll[1];
        } else {
          lon = srcX;  lat = srcY;
        }

        // Elevation and color
        let h = elev ? Number(elev[i]) : 0;
        let r = hasRGB ? Number(R[i]) : 200;
        let g = hasRGB ? Number(G[i]) : 200;
        let b = hasRGB ? Number(B[i]) : 200;
        let a = 255;

        if (h === NODATA_ELEV || h === 0) {
          // Treat as "no elev" → hole; keep color light/transparent if you visualize point cloud
          h = 0;
          a = 100;
          r = 255; g = 255; b = 255;
          // We SKIP adding this vertex to the shared grid (keeps holes so world terrain shows through)
          continue;
        }

        h += Number(deltaZ || 0);

        positions.push(lat, lon, h);   // store as [lat, lon, h]
        colors.push(r, g, b, a);
        indexMap[i] = nextIdx++;
      }
    }

    const posArr = new Float64Array(positions);
    const colArr = new Uint8Array(colors);

    // Smooth normals on SHARED grid (two-tri per cell, shared vertices)
    const idxShared = buildSharedGridIndices(indexMap, width, height);
    const sharedNormals = computeNormals(posArr, idxShared);

    // Emit center-fan with flat per-cell color but smooth normals copied in
    const { outPositions, outColors, outIndices, outNormals } =
      buildCenterFanWithSmoothNormals(posArr, colArr, sharedNormals, indexMap, width, height);

    // Also provide a WGS84 bbox for convenience if source was Web Mercator
    let bboxWGS84 = null;
    if (isWM) {
      const llMin = webMercatorToLonLat(minX, minY);
      const llMax = webMercatorToLonLat(maxX, maxY);
      bboxWGS84 = [llMin[0], llMin[1], llMax[0], llMax[1]];
    }

    self.postMessage({
      id,
      ok: true,
      result: {
        width,
        height,
        bboxSource: bboxSrc,      // original image bbox (source CRS)
        bboxWGS84: bboxWGS84,     // filled if source was 3857
        positions: outPositions.buffer, // Float64Array
        colors:    outColors.buffer,    // Uint8Array
        indices:   outIndices.buffer,   // Uint32Array
        normals:   outNormals.buffer    // Float32Array
      }
    }, [outPositions.buffer, outColors.buffer, outIndices.buffer, outNormals.buffer]);

  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: (err && err.message) || String(err)
    });
  }
};
