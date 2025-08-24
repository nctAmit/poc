/* meshWorker.js */
/* CPU-bound pipeline:
   - Parse GeoTIFF
   - Build positions/colors/indices (solid quads → two triangles)
   - Compute normals
   - Bake hillshade + optional contours into colors so subtle elevation is visible
   Returns typed arrays via Transferables
*/

self.importScripts('https://cdn.jsdelivr.net/npm/geotiff');

const NODATA_ELEV = -99999;

// --------- Utils ----------
function clamp01(v) { return Math.max(0, Math.min(1, v)); }
function clamp255(v) { return Math.max(0, Math.min(255, v | 0)); }
function degToRad(d) { return d * Math.PI / 180; }

// Lambertian hillshade from ENU normal and light dir
function hillshadeFromNormal(nx, ny, nz, Lx, Ly, Lz) {
  const d = nx * Lx + ny * Ly + nz * Lz;
  // Lambert clamp with soft ambient so fully-backfacing isn’t pure black
  return 0.25 + 0.75 * clamp01(d);
}

// ----- Normals (per-vertex, accumulated from triangle faces) -----
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

// Approx hillshade on the source grid (in meters) using central differences.
// We do this BEFORE meshing so a flat color tile still gets baked relief.
function computeGridHillshade(elev, width, height, minLon, maxLon, minLat, maxLat, opts) {
  const { zExaggeration = 1, sunAzimuthDeg = 315, sunAltitudeDeg = 45 } = opts || {};

  const out = new Float32Array(width * height);

  // meters per degree (approx) – vary lon scale by latitude to reduce distortion
  const latMid = (minLat + maxLat) * 0.5;
  const mPerDegLat = 110540; // good average
  const mPerDegLon = 111320 * Math.cos(degToRad(latMid));

  const dLon = (maxLon - minLon) / (width - 1 || 1);
  const dLat = (maxLat - minLat) / (height - 1 || 1);
  const dx = Math.abs(dLon) * mPerDegLon;
  const dy = Math.abs(dLat) * mPerDegLat;

  // Light in local ENU (x east, y north, z up)
  const az = degToRad(sunAzimuthDeg);     // 0 = North, 90 = East
  const alt = degToRad(sunAltitudeDeg);   // above horizon
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

      // ENU normal from gradient
      let nx = -dzdx, ny = -dzdy, nz = 1;
      const invLen = 1 / Math.hypot(nx, ny, nz);
      nx *= invLen; ny *= invLen; nz *= invLen;

      out[idx(x, y)] = hillshadeFromNormal(nx, ny, nz, Lx, Ly, Lz); // 0..1
    }
  }
  return out;
}

// Build “solid rectangles” (grid cell → 2 triangles) with single color per cell (from top-left),
// BUT we bake hillshade (and optional contours) into the color so subtleties are visible.
function buildSolidRectGrid(grid, positionsAll, colorsAll, indexMap, width, height, hillshadeGrid, elevGrid, opts) {
  const {
    shadeStrength = 0.6,                        // 0..1 how much to blend hillshade
    contourInterval = 0,                         // meters, 0 to disable
    contourThickness = 0.25,                     // meters – “half-width” of band around interval
    contourStrength = 0.6,                       // 0..1 darkening strength for contours
  } = opts || {};

  // Count quads
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

  const applyShading = (r, g, b, hs /*0..1*/) => {
    // Blend original color towards hillshade gray value.
    // Grayscale target = hs * 255
    const lit = hs * 255;
    const s = clamp01(shadeStrength);
    const nr = clamp255((1 - s) * r + s * lit);
    const ng = clamp255((1 - s) * g + s * lit);
    const nb = clamp255((1 - s) * b + s * lit);
    return [nr, ng, nb];
  };

  const contourDarken = (r, g, b) => {
    const s = clamp01(contourStrength);
    const k = 1 - s;
    return [clamp255(r * k), clamp255(g * k), clamp255(b * k)];
  };

  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x;
      const tr = tl + 1;
      const bl = (y + 1) * width + x;
      const br = bl + 1;

      const i0 = indexMap[tl], i1 = indexMap[bl], i2 = indexMap[br], i3 = indexMap[tr];
      if (i0 === -1 || i1 === -1 || i2 === -1 || i3 === -1) continue;

      // Base color from top-left vertex of the cell
      const cBase = i0 * 4;
      let r = colorsAll[cBase], g = colorsAll[cBase + 1], b = colorsAll[cBase + 2], a = colorsAll[cBase + 3];

      // Shade from grid hillshade at TL
      const hs = hillshadeGrid ? hillshadeGrid[tl] : 0.5;
      [r, g, b] = applyShading(r, g, b, hs);

      // Optional contours based on elevation at TL (in Z units = meters after exaggeration in upstream step)
      if (contourInterval > 0) {
        const h = elevGrid ? elevGrid[tl] : 0;
        const t = Math.abs((h % contourInterval + contourInterval) % contourInterval); // wrap positive
        if (t <= contourThickness || (contourInterval - t) <= contourThickness) {
          [r, g, b] = contourDarken(r, g, b);
        }
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
        outColors[co + 3] = a;
        vtx++;
      };

      addCopy(i0);
      addCopy(i1);
      addCopy(i2);
      addCopy(i3);

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
      // New tuning knobs (all optional)
      zExaggeration = 1.0,         // multiply Z to make small relief visible
      shadeStrength = 0.6,         // 0..1 – how much hillshade to bake into colors
      sunAzimuthDeg = 315,         // 0=N, 90=E (same convention as ArcGIS hillshade)
      sunAltitudeDeg = 45,         // angle above horizon
      contourInterval = 0,         // meters (0 disables)
      contourThickness = 0.25,     // meters “half width” of the line
      contourStrength = 0.6,       // 0..1 darken factor for contour marks
    } = payload || {};

    const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    const image = await tiff.getImage();
    const rasters = await image.readRasters(); // [0]=elev, [1]=R, [2]=G, [3]=B
    const width = image.getWidth();
    const height = image.getHeight();

    // If GeoTIFF has GeoKeys, these are in its CRS; for EPSG:4326 it’s degrees.
    const bbox = image.getBoundingBox(); // [minX, minY, maxX, maxY]
    const minX = bbox[0], minY = bbox[1], maxX = bbox[2], maxY = bbox[3];
    const lonStep = (maxX - minX) / (width - 1 || 1);
    const latStep = (maxY - minY) / (height - 1 || 1);

    const elev = rasters[0];
    const R = rasters[1], G = rasters[2], B = rasters[3];

    // Precompute hillshade per source pixel (works even when colors are flat)
    const hsGrid = computeGridHillshade(
      elev, width, height, minX, maxX, minY, maxY,
      { zExaggeration, sunAzimuthDeg, sunAltitudeDeg }
    );

    // Build sparse vertex map (skip nodata / zero elev -> transparent)
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
          a = 100; r = 255; g = 255; b = 255; // translucent white “no data”
        } else {
          h = (h + Number(deltaZ || 0)) * Number(zExaggeration || 1);
        }

        if (h > 0) {
          // Positions remain (lat, lon, height) to keep parity with your main thread
          positions.push(lat, lon, h);
          colors.push(r, g, b, a);
          indexMap[i] = nextIdx++;
        }
      }
    }

    const posArr = new Float64Array(positions);
    const colArr = new Uint8Array(colors);

    // Make quads, baking in hillshade + optional contours into the colors
    const { outPositions, outColors, outIndices } =
      buildSolidRectGrid({ width, height }, posArr, colArr, indexMap, width, height, hsGrid, elev, {
        shadeStrength, contourInterval, contourThickness, contourStrength,
      });

    // Mesh normals (for real lighting if your material uses it)
    const normals = computeNormals(outPositions, outIndices);

    // Return
    self.postMessage({
      id,
      ok: true,
      result: {
        width,
        height,
        bbox, // [minX, minY, maxX, maxY]
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
