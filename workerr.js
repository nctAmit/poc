/* meshWorker.js */
/* Pipeline:
   - Parse GeoTIFF
   - Build center-fan triangles per cell: (C, TL, BL), (C, BL, BR), (C, BR, TR), (C, TR, TL)
   - Per-triangle FLAT color (same for all 3 vertices)
   - Per-triangle FLAT normal (same for all 3 vertices), computed in local ENU (meters)
   - Transfer typed arrays
*/

self.importScripts('https://cdn.jsdelivr.net/npm/geotiff');

const NODATA_ELEV = -99999;

// ---------- CRS helpers ----------
function bboxIsWebMercator(bbox) {
  const [minX, minY, maxX, maxY] = bbox;
  const maxAbs = Math.max(Math.abs(minX), Math.abs(minY), Math.abs(maxX), Math.abs(maxY));
  return maxAbs > 360; // numbers >> degrees → meters (EPSG:3857)
}

function webMercatorToLonLat(x, y) {
  const R = 6378137.0;
  const lon = (x / R) * (180 / Math.PI);
  const lat = (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) * (180 / Math.PI);
  return [lon, lat];
}

// ---------- ENU (meters) relative to tile center ----------
function makeLonLatToENU(lon0, lat0) {
  const R = 6378137.0;
  const lat0rad = lat0 * Math.PI / 180;
  const cosLat0 = Math.cos(lat0rad);
  return function lonLatH_to_ENU(lon, lat, h) {
    const xEast  = (lon - lon0) * (Math.PI / 180) * R * cosLat0;
    const yNorth = (lat - lat0) * (Math.PI / 180) * R;
    const zUp    = h;
    return [xEast, yNorth, zUp];
  };
}

// ---------- vector helpers ----------
function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function normalize(v) {
  const L = Math.hypot(v[0], v[1], v[2]);
  return L > 0 ? [v[0]/L, v[1]/L, v[2]/L] : [0,0,1];
}

// ---------- core builder: center-connected triangles with flat colors & face normals ----------
function buildCenterFanFlatTri(positionsGrid /* Float64Array [lat,lon,h]* */,
                               colorsGrid    /* Uint8Array [r,g,b,a]* */,
                               indexMap      /* Int32Array */,
                               width, height,
                               lon0, lat0) {

  // Pre-bind ENU converter
  const toENU = makeLonLatToENU(lon0, lat0);

  // Count valid quads (all 4 corners exist)
  let quadCount = 0;
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x;
      const tr = tl + 1;
      const bl = (y + 1) * width + x;
      const br = bl + 1;
      if (indexMap[tl] !== -1 && indexMap[tr] !== -1 && indexMap[bl] !== -1 && indexMap[br] !== -1) {
        quadCount++;
      }
    }
  }

  // 4 triangles per quad, 3 vertices per triangle → 12 vertices per quad (no sharing — flat color & flat normal)
  const TRIS_PER_QUAD = 4;
  const VERTS_PER_TRI = 3;
  const VERTS_PER_QUAD = TRIS_PER_QUAD * VERTS_PER_TRI; // 12

  const outPositions = new Float64Array(quadCount * VERTS_PER_QUAD * 3);
  const outColors    = new Uint8Array (quadCount * VERTS_PER_QUAD * 4);
  const outNormals   = new Float32Array(quadCount * VERTS_PER_QUAD * 3);
  const outIndices   = new Uint32Array(quadCount * TRIS_PER_QUAD  * 3);

  let vtx = 0;
  let idx = 0;

  // Helper to read one grid vertex (lat,lon,h) & color from shared arrays by grid index
  function getPos(i) {
    const p = i * 3;
    return [positionsGrid[p], positionsGrid[p+1], positionsGrid[p+2]]; // [lat, lon, h]
  }
  function getColor(i) {
    const c = i * 4;
    return [colorsGrid[c], colorsGrid[c+1], colorsGrid[c+2], colorsGrid[c+3]];
  }

  // Push one triangle with FLAT color & FLAT normal (duplicate 3 vertices)
  function pushFlatTriangle(Pa, Pb, Pc, color) {
    // Compute face normal in ENU
    const aE = toENU(Pa[1], Pa[0], Pa[2]); // [lon,lat,h] → ENU; note Pa = [lat,lon,h]
    const bE = toENU(Pb[1], Pb[0], Pb[2]);
    const cE = toENU(Pc[1], Pc[0], Pc[2]);
    const e1 = sub(bE, aE);
    const e2 = sub(cE, aE);
    const n  = normalize(cross(e1, e2));

    // Write 3 duplicated vertices (positions in [lat,lon,h] to match your upstream)
    for (const P of [Pa, Pb, Pc]) {
      const o3 = vtx * 3;
      const o4 = vtx * 4;

      outPositions[o3]     = P[0];
      outPositions[o3 + 1] = P[1];
      outPositions[o3 + 2] = P[2];

      outNormals[o3]       = n[0];
      outNormals[o3 + 1]   = n[1];
      outNormals[o3 + 2]   = n[2];

      outColors[o4]        = color[0];
      outColors[o4 + 1]    = color[1];
      outColors[o4 + 2]    = color[2];
      outColors[o4 + 3]    = color[3];

      outIndices[idx++] = vtx;
      vtx++;
    }
  }

  // Build triangles
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const tl = y * width + x;
      const tr = tl + 1;
      const bl = (y + 1) * width + x;
      const br = bl + 1;

      const iTL = indexMap[tl], iTR = indexMap[tr], iBL = indexMap[bl], iBR = indexMap[br];
      if (iTL === -1 || iTR === -1 || iBL === -1 || iBR === -1) continue;

      // Corners
      const PTL = getPos(iTL);
      const PTR = getPos(iTR);
      const PBL = getPos(iBL);
      const PBR = getPos(iBR);

      // Center = average of 4 corners in [lat,lon,h]
      const C = [
        (PTL[0] + PTR[0] + PBL[0] + PBR[0]) * 0.25,
        (PTL[1] + PTR[1] + PBL[1] + PBR[1]) * 0.25,
        (PTL[2] + PTR[2] + PBL[2] + PBR[2]) * 0.25
      ];

      // Flat color per cell from TL (keep your existing look)
      const color = getColor(iTL);

      // Fan triangles (counter-clockwise for consistent normals):
      // (C, TL, BL), (C, BL, BR), (C, BR, TR), (C, TR, TL)
      pushFlatTriangle(C, PTL, PBL, color);
      pushFlatTriangle(C, PBL, PBR, color);
      pushFlatTriangle(C, PBR, PTR, color);
      pushFlatTriangle(C, PTR, PTL, color);
    }
  }

  return { outPositions, outColors, outIndices, outNormals };
}

// ---------- worker ----------
self.onmessage = async (e) => {
  const { id, type, payload } = e.data || {};
  try {
    if (type !== 'PROCESS_TIFF') throw new Error(`Unknown worker message type: ${type}`);
    const { arrayBuffer, deltaZ = 0 } = payload;

    const tiff    = await GeoTIFF.fromArrayBuffer(arrayBuffer);
    const image   = await tiff.getImage();
    const rasters = await image.readRasters(); // [0]=elev, [1]=R, [2]=G, [3]=B (if present)
    const width   = image.getWidth();
    const height  = image.getHeight();

    const bboxSrc = image.getBoundingBox(); // [minX, minY, maxX, maxY] in source CRS
    const isWM    = bboxIsWebMercator(bboxSrc);

    const minX = bboxSrc[0], minY = bboxSrc[1], maxX = bboxSrc[2], maxY = bboxSrc[3];
    const stepX = (maxX - minX) / (width  - 1 || 1);
    const stepY = (maxY - minY) / (height - 1 || 1);

    const elev = rasters[0];
    const hasRGB = rasters.length >= 4;
    const R = hasRGB ? rasters[1] : null;
    const G = hasRGB ? rasters[2] : null;
    const B = hasRGB ? rasters[3] : null;

    // Shared grid: store only valid samples; indexMap maps image grid -> dense arrays
    const positions = []; // [lat,lon,h]
    const colors    = []; // [r,g,b,a]
    const indexMap  = new Int32Array(width * height);
    indexMap.fill(-1);

    // Tile center in lon/lat (used only for ENU normal calc)
    let lon0, lat0;
    if (isWM) {
      const midX = 0.5 * (minX + maxX);
      const midY = 0.5 * (minY + maxY);
      [lon0, lat0] = webMercatorToLonLat(midX, midY);
    } else {
      lon0 = 0.5 * (minX + maxX);
      lat0 = 0.5 * (minY + maxY);
    }

    let nextIdx = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        // source coords → lon/lat
        const srcX = minX + x * stepX;
        const srcY = maxY - y * stepY;
        let lon, lat;
        if (isWM) {
          [lon, lat] = webMercatorToLonLat(srcX, srcY);
        } else {
          lon = srcX; lat = srcY;
        }

        // elevation & color
        let h = elev ? Number(elev[i]) : 0;
        if (h === NODATA_ELEV || h === 0) {
          // skip holes so base terrain can show through
          continue;
        }
        h += Number(deltaZ || 0);

        const r = hasRGB ? Number(R[i]) : 200;
        const g = hasRGB ? Number(G[i]) : 200;
        const b = hasRGB ? Number(B[i]) : 200;
        const a = 255;

        positions.push(lat, lon, h);     // store as [lat,lon,h]
        colors.push(r, g, b, a);
        indexMap[i] = nextIdx++;
      }
    }

    const posArr = new Float64Array(positions);
    const colArr = new Uint8Array(colors);

    // Build center-connected triangles with flat color & face normals
    const { outPositions, outColors, outIndices, outNormals } =
      buildCenterFanFlatTri(posArr, colArr, indexMap, width, height, lon0, lat0);

    // Optional: provide WGS84 bbox when source was Web Mercator
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
        bboxSource: bboxSrc,
        bboxWGS84,
        positions: outPositions.buffer, // Float64Array
        colors:    outColors.buffer,    // Uint8Array
        indices:   outIndices.buffer,   // Uint32Array
        normals:   outNormals.buffer    // Float32Array
      }
    }, [outPositions.buffer, outColors.buffer, outIndices.buffer, outNormals.buffer]);

  } catch (err) {
    self.postMessage({ id, ok: false, error: (err && err.message) || String(err) });
  }
};
