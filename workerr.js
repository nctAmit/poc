/* meshWorker.js */
/* CPU-bound pipeline:
   - Parse GeoTIFF
   - Build positions/colors/indices (solid quads → two triangles)
   - Compute normals
   Returns typed arrays via Transferables
*/

self.importScripts('https://cdn.jsdelivr.net/npm/geotiff');

const NODATA_ELEV = -99999;

// ----- WGS84 to ECEF (so we don’t need Cesium in worker) -----
function degToRad(d) { return d * Math.PI / 180; }

// ----- Normals (per-vertex, accumulated from triangle faces) -----
function computeNormals(positions /* Float64Array */, indices /* Uint32Array */) {
  const n = positions.length / 3;
  const normals = new Float32Array(positions.length); // 3 per vertex
  // zero init by default

  for (let i = 0; i < indices.length; i += 3) {
    const ia = indices[i] * 3;
    const ib = indices[i + 1] * 3;
    const ic = indices[i + 2] * 3;

    const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2];
    const bx = positions[ib], by = positions[ib + 1], bz = positions[ib + 2];
    const cx = positions[ic], cy = positions[ic + 1], cz = positions[ic + 2];

    // edges
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;

    // face normal = e1 x e2
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;

    normals[ia] += nx; normals[ia + 1] += ny; normals[ia + 2] += nz;
    normals[ib] += nx; normals[ib + 1] += ny; normals[ib + 2] += nz;
    normals[ic] += nx; normals[ic + 1] += ny; normals[ic + 2] += nz;
  }

  // normalize
  for (let i = 0; i < normals.length; i += 3) {
    const nx = normals[i], ny = normals[i + 1], nz = normals[i + 2];
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) {
      normals[i] = nx / len;
      normals[i + 1] = ny / len;
      normals[i + 2] = nz / len;
    }
  }
  return normals;
}

// Build “solid rectangles” (grid cell → 2 triangles) with single color per cell (from top-left)
function buildSolidRectGrid(grid, positionsAll, colorsAll, indexMap, width, height) {
  // Count quads
  let quadCount = 0;
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x;
      const tr = tl + 1;
      const bl = (y + 1) * width + x;
      const br = bl + 1;
      if (
        indexMap[tl] !== -1 &&
        indexMap[tr] !== -1 &&
        indexMap[bl] !== -1 &&
        indexMap[br] !== -1
      ) quadCount++;
    }
  }

  const outPositions = new Float64Array(quadCount * 4 * 3); // 4 verts * 3
  const outColors = new Uint8Array(quadCount * 4 * 4);   // 4 verts * RGBA
  const outIndices = new Uint32Array(quadCount * 6);      // 2 tris * 3

  let vtx = 0;  // vertex cursor (4 per quad)
  let idx = 0;  // index cursor (6 per quad)

  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x;
      const tr = tl + 1;
      const bl = (y + 1) * width + x;
      const br = bl + 1;

      const i0 = indexMap[tl];
      const i1 = indexMap[bl];
      const i2 = indexMap[br];
      const i3 = indexMap[tr];

      if (i0 === -1 || i1 === -1 || i2 === -1 || i3 === -1) continue;

      // Color from top-left
      const cBase = i0 * 4;
      const r = colorsAll[cBase], g = colorsAll[cBase + 1], b = colorsAll[cBase + 2], a = colorsAll[cBase + 3];

      const addCopy = (srcIndex) => {
        const pBase = srcIndex * 3;
        const o = vtx * 3;
        outPositions[o] = positionsAll[pBase];
        outPositions[o + 1] = positionsAll[pBase + 1];
        outPositions[o + 2] = positionsAll[pBase + 2];

        const co = vtx * 4;
        outColors[co] = r;
        outColors[co + 1] = g;
        outColors[co + 2] = b;
        outColors[co + 3] = a;
        vtx++;
      };

      addCopy(i0);
      addCopy(i1);
      addCopy(i2);
      addCopy(i3);

      // two triangles: (0,1,2) and (0,2,3) within the 4 new vertices of this quad
      const base = vtx - 4;
      outIndices[idx++] = base;
      outIndices[idx++] = base + 1;
      outIndices[idx++] = base + 2;
      outIndices[idx++] = base;
      outIndices[idx++] = base + 2;
      outIndices[idx++] = base + 3;
    }
  }

  return { outPositions, outColors, outIndices };
}

function webMercatorToLonLat(x, y) {
  const R = 6378137.0;
  const lon = (x / R) * (180 / Math.PI);
  const lat = (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) * (180 / Math.PI);
  return [lon, lat];
}

self.onmessage = async (e) => {
  const { id, type, payload } = e.data || {};
  try {
    if (type === 'PROCESS_TIFF') {
      const { arrayBuffer, deltaZ = 0 } = payload;

      const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
      const image = await tiff.getImage();
      const rasters = await image.readRasters(); // [0]=elev, [1]=R, [2]=G, [3]=B (as in your pipeline)
      const width = image.getWidth();
      const height = image.getHeight();

      const bbox = image.getBoundingBox(); // [minX, minY, maxX, maxY]
      const minX = bbox[0], minY = bbox[1], maxX = bbox[2], maxY = bbox[3];
      const lonStep = (maxX - minX) / (width - 1 || 1);
      const latStep = (maxY - minY) / (height - 1 || 1);

      const elev = rasters[0];
      const R = rasters[1], G = rasters[2], B = rasters[3];

      // Build sparse vertex map (skip nodata / zero elev)
      const positions = [];
      const colors = [];
      const indexMap = new Int32Array(width * height);
      indexMap.fill(-1);

      let nextIdx = 0;

      for (let y = 0; y < height; y++) {
        const lat = maxY - y * latStep;
        for (let x = 0; x < width; x++) {
          const lon = minX + x * lonStep;
          const i = y * width + x;

          let h = (elev[i] === NODATA_ELEV) ? 0 : Number(elev[i]);
          let r = Number(R[i]), g = Number(G[i]), b = Number(B[i]);
          let a = 255;

          if (h === 0) {
            // transparent white for no-elev
            h = 0;
            a = 100;
            r = 255; g = 255; b = 255;
          } else {
            h += Number(deltaZ || 0);
          }

          if (h > 0) {
            positions.push(lat, lon, h);
            colors.push(r, g, b, a);
            indexMap[i] = nextIdx++;
          }
        }
      }

      const posArr = new Float64Array(positions);
      const colArr = new Uint8Array(colors);

      // Solid rectangles (quads)
      const { outPositions, outColors, outIndices } =
        buildSolidRectGrid({ width, height }, posArr, colArr, indexMap, width, height);

      const normals = computeNormals(outPositions, outIndices);

      // Prepare result (use transferables to avoid copying)
      self.postMessage({
        id,
        ok: true,
        result: {
          width,
          height,
          bbox, // [minX, minY, maxX, maxY] (degrees if source is EPSG:4326)
          positions: outPositions.buffer,
          colors: outColors.buffer,
          indices: outIndices.buffer,
          normals: normals.buffer
        }
      }, [outPositions.buffer, outColors.buffer, outIndices.buffer, normals.buffer]);

    } else {
      throw new Error(`Unknown worker message type: ${type}`);
    }
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: (err && err.message) || String(err)
    });
  }
};