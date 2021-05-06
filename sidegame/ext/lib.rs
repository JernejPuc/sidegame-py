use ndarray::{Ix1, Ix2, Array, ArrayView};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

/// # SDGlib -- Extensions for the "Simplified defusal game" (SiDeGame/SDG)
/// `sdglib` is used mainly to perform pixel iteration on a lower level.
#[pymodule]
fn sdglib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn move_player(
        self_id: i16,
        pos_1: ArrayView<f64, Ix1>,
        pos_2: ArrayView<f64, Ix1>,
        height_map: ArrayView<u8, Ix2>,
        player_id_map: ArrayView<i16, Ix2>
    ) -> Array<f64, Ix1> {

        let terrain_transition: u8 = 63;
        let terrain_elevated: u8 = 127;
        let player_id_null: i16 = 32767;

        let (mut terrain, mut player_id): (u8, i16);

        let mut in_transition: bool = false;
        let mut collision_detected: bool = false;

        // Round to pixel positions
        let (x0, y0): (i64, i64) = (pos_1[0].round() as i64, pos_1[1].round() as i64);
        let (x1, y1): (i64, i64) = (pos_2[0].round() as i64, pos_2[1].round() as i64);

        // Init out
        let mut pos_2_check: Array<f64, Ix1> = Array::zeros(2);

        // Init for Bresenham
        let dx: i64 = (x1 - x0).abs();
        let dy: i64 = -(y1 - y0).abs();

        let mut e: i64 = dx + dy;
        let mut e2: i64;

        let (mut ty, mut tx): (i64, i64) = (y0, x0);
        let (mut ty_prev, mut tx_prev): (i64, i64) = (ty, tx);

        let (sx, sy): (i64, i64);

        if x0 < x1 {
            sx = 1;
        } else {
            sx = -1;
        }

        if y0 < y1 {
            sy = 1;
        } else {
            sy = -1;
        }

        // Check 9 core pixels (3x3) for all possibilities
        for i in -1..=1 {
            for j in -1..=1 {
                terrain = height_map[[(ty+i) as usize, (tx+j) as usize]];

                if terrain >= terrain_transition {
                    in_transition = true;
                }
            }
        }

        // Trace up to target or collision
        loop {
            for i in -1..=1 {
                for j in -1..=1 {
                    terrain = height_map[[(ty+i) as usize, (tx+j) as usize]];
                    player_id = player_id_map[[(ty+i) as usize, (tx+j) as usize]];

                    if ((player_id != player_id_null) & (player_id != self_id))
                    | (terrain > terrain_elevated)
                    | ((terrain == terrain_elevated) & (!in_transition)) {
                        collision_detected = true;
                    }
                }
            }

            if collision_detected {
                pos_2_check[0] = tx_prev as f64;
                pos_2_check[1] = ty_prev as f64;
                break;
            } else {
                tx_prev = tx;
                ty_prev = ty;
            }

            // Bresenham
            if (tx == x1) & (ty == y1) {
                break;
            }

            e2 = 2*e;

            if e2 >= dy {
                e += dy;
                tx += sx;
            }

            if e2 <= dx {
                e += dx;
                ty += sy;
            }
        }

        pos_2_check
    }

    fn move_object(
        self_id: i16,
        pos_1: ArrayView<f64, Ix1>,
        pos_2: ArrayView<f64, Ix1>,
        wall_map: ArrayView<u8, Ix2>,
        object_map: ArrayView<i16, Ix2>
    ) -> Array<f64, Ix1> {

        let terrain_wall: u8 = 1;
        let object_id_null: i16 = 0;

        let (mut terrain, mut object): (u8, i16);

        let mut collision_detected: bool = false;

        // Round to pixel positions
        let (x0, y0): (i64, i64) = (pos_1[0].round() as i64, pos_1[1].round() as i64);
        let (x1, y1): (i64, i64) = (pos_2[0].round() as i64, pos_2[1].round() as i64);

        // Init out
        let mut pos_2_check: Array<f64, Ix1> = Array::zeros(2);

        // Init for Bresenham
        let dx: i64 = (x1 - x0).abs();
        let dy: i64 = -(y1 - y0).abs();

        let mut e: i64 = dx + dy;
        let mut e2: i64;

        let (mut ty, mut tx) = (y0, x0);
        let (mut ty_prev, mut tx_prev) = (ty, tx);

        let (sx, sy): (i64, i64);

        if x0 < x1 {
            sx = 1;
        } else {
            sx = -1;
        }

        if y0 < y1 {
            sy = 1;
        } else {
            sy = -1;
        }

        // Trace up to target or collision
        loop {
            terrain = wall_map[[ty as usize, tx as usize]];
            object = object_map[[ty as usize, tx as usize]];

            if (terrain == terrain_wall)
            | ((object != object_id_null) & (object != self_id)) {
                collision_detected = true;
            }

            if collision_detected {
                pos_2_check[0] = tx_prev as f64;
                pos_2_check[1] = ty_prev as f64;
                break;
            } else {
                tx_prev = tx;
                ty_prev = ty;
            }

            // Bresenham
            if (tx == x1) & (ty == y1) {
                break;
            }

            e2 = 2*e;

            if e2 >= dy {
                e += dy;
                tx += sx;
            }

            if e2 <= dx {
                e += dx;
                ty += sy;
            }
        }

        pos_2_check
    }

    fn trace_shot(
        self_id: i16,
        pos_1: ArrayView<f64, Ix1>,
        pos_2: ArrayView<f64, Ix1>,
        height_map: ArrayView<u8, Ix2>,
        player_id_map: ArrayView<i16, Ix2>
    ) -> Array<f64, Ix1> {

        let terrain_elevated: u8 = 127;
        let player_id_null: i16 = 32767;

        let (mut terrain, mut player_id): (u8, i16);

        // Round to pixel positions
        let (x0, y0): (i64, i64) = (pos_1[0].round() as i64, pos_1[1].round() as i64);
        let (x1, y1): (i64, i64) = (pos_2[0].round() as i64, pos_2[1].round() as i64);

        // Init out
        let mut pos_2_check: Array<f64, Ix1> = Array::zeros(2);

        // Init for bresenham
        let dx: i64 = (x1 - x0).abs();
        let dy: i64 = -(y1 - y0).abs();

        let mut e: i64 = dx + dy;
        let mut e2: i64;

        let (mut ty, mut tx): (i64, i64) = (y0, x0);
        let (sx, sy): (i64, i64);

        if x0 < x1 {
            sx = 1;
        } else {
            sx = -1;
        }

        if y0 < y1 {
            sy = 1;
        } else {
            sy = -1;
        }

        // Trace up to hit or end of range
        loop {
            terrain = height_map[[ty as usize, tx as usize]];
            player_id = player_id_map[[ty as usize, tx as usize]];

            if ((player_id != player_id_null) & (player_id != self_id))
            | (terrain > terrain_elevated) {
                pos_2_check[0] = tx as f64;
                pos_2_check[1] = ty as f64;
                break;
            }

            // Bresenham
            if (tx == x1) & (ty == y1) {
                break;
            }

            e2 = 2*e;

            if e2 >= dy {
                e += dy;
                tx += sx;
            }

            if e2 <= dx {
                e += dx;
                ty += sy;
            }
        }

        pos_2_check
    }

    fn trace_sight(
        self_id: i16,
        pos_1: ArrayView<f64, Ix1>,
        pos_2: ArrayView<f64, Ix1>,
        height_map: ArrayView<u8, Ix2>,
        player_id_map: ArrayView<i16, Ix2>,
        zone_map: ArrayView<u8, Ix2>
    ) -> Array<f64, Ix1> {

        let terrain_elevated: u8 = 127;
        let player_id_null: i16 = 32767;
        let zone_smoke: u8 = 255;

        let (mut terrain, mut player_id, mut zone): (u8, i16, u8);

        // Round to pixel positions
        let (x0, y0): (i64, i64) = (pos_1[0].round() as i64, pos_1[1].round() as i64);
        let (x1, y1): (i64, i64) = (pos_2[0].round() as i64, pos_2[1].round() as i64);

        // Init out
        let mut pos_2_check: Array<f64, Ix1> = Array::zeros(2);

        // Init for Bresenham
        let dx: i64 = (x1 - x0).abs();
        let dy: i64 = -(y1 - y0).abs();

        let mut e: i64 = dx + dy;
        let mut e2: i64;

        let (mut ty, mut tx): (i64, i64) = (y0, x0);
        let (sx, sy): (i64, i64);

        if x0 < x1 {
            sx = 1;
        } else {
            sx = -1;
        }

        if y0 < y1 {
            sy = 1;
        } else {
            sy = -1;
        }

        // Trace up to target or occlusion
        loop {
            terrain = height_map[[ty as usize, tx as usize]];
            player_id = player_id_map[[ty as usize, tx as usize]];
            zone = zone_map[[ty as usize, tx as usize]];

            if ((player_id != player_id_null) & (player_id != self_id))
            | (zone == zone_smoke)
            | (terrain > terrain_elevated) {
                pos_2_check[0] = tx as f64;
                pos_2_check[1] = ty as f64;
                break;
            }

            // Bresenham
            if (tx == x1) & (ty == y1) {
                break;
            }

            e2 = 2*e;

            if e2 >= dy {
                e += dy;
                tx += sx;
            }

            if e2 <= dx {
                e += dx;
                ty += sy;
            }
        }

        pos_2_check
    }

    fn mask_visible_line(
        self_id: i16,
        y0: i64,
        x0: i64,
        y1: i64,
        x1: i64,
        hmap: ArrayView<u8, Ix2>,
        emap: ArrayView<i16, Ix2>,
        fmap: ArrayView<u8, Ix2>,
        mut mask: Array<u8, Ix2>
    ) -> Array<u8, Ix2> {

        let terrain_elevated: u8 = 127;
        let player_id_null: i16 = 32767;
        let zone_smoke: u8 = 255;

        let (mut terrain, mut player_id, mut zone): (u8, i16, u8);

        // Init for Bresenham
        let dx: i64 = (x1 - x0).abs();
        let dy: i64 = -(y1 - y0).abs();

        let mut e: i64 = dx + dy;
        let mut e2: i64;

        let (mut ty, mut tx): (i64, i64) = (y0, x0);
        let (sx, sy): (i64, i64);

        if x0 < x1 {
            sx = 1;
        } else {
            sx = -1;
        }
        
        if y0 < y1 {
            sy = 1;
        } else {
            sy = -1;
        }

        // Mask up to endpoint or occlusion
        loop {
            terrain = hmap[[ty as usize, tx as usize]];
            player_id = emap[[ty as usize, tx as usize]];
            zone = fmap[[ty as usize, tx as usize]];

            if (terrain > terrain_elevated)
            | (zone == zone_smoke)
            | ((player_id != player_id_null) & (player_id != self_id)) {
                break;
            }

            mask[[ty as usize, tx as usize]] = 1;

            // Bresenham
            if (tx == x1) & (ty == y1) {
                break;
            }

            e2 = 2*e;

            if e2 >= dy {
                e += dy;
                tx += sx;
            }

            if e2 <= dx {
                e += dx;
                ty += sy;
            }
        }

        mask
    }

    fn mask_visible(
        self_id: i16,
        height_map: ArrayView<u8, Ix2>,
        player_map: ArrayView<i16, Ix2>,
        zone_map: ArrayView<u8, Ix2>,
        left_ends_y: ArrayView<i64, Ix1>,
        left_ends_x: ArrayView<i64, Ix1>,
        right_ends_y: ArrayView<i64, Ix1>,
        right_ends_x: ArrayView<i64, Ix1>
    ) -> Array<u8, Ix2> {

        let (y0, x0_left, x0_right): (i64, i64, i64) = (107, 95, 96);
        let (mut x1, mut y1): (i64, i64);

        let mut mask: Array<u8, Ix2> = Array::zeros((108, 192));

        for idx in 0..left_ends_y.shape()[0] {
            x1 = left_ends_x[idx];
            y1 = left_ends_y[idx];

            mask = mask_visible_line(self_id, y0, x0_left, y1, x1, height_map, player_map, zone_map, mask);
        }

        for idx in 0..right_ends_y.shape()[0] {
            x1 = right_ends_x[idx];
            y1 = right_ends_y[idx];

            mask = mask_visible_line(self_id, y0, x0_right, y1, x1, height_map, player_map, zone_map, mask);
        }

        mask
    }

    fn mask_ray(
        length: i16,
        pos_1: ArrayView<f64, Ix1>,
        pos_2: ArrayView<f64, Ix1>
    ) -> Array<u8, Ix2> {

        // Round to pixel positions
        let (x0, y0): (i64, i64) = (pos_1[0].round() as i64, pos_1[1].round() as i64);
        let (x1, y1): (i64, i64) = (pos_2[0].round() as i64, pos_2[1].round() as i64);

        // Init out
        let mut mask: Array<u8, Ix2> = Array::zeros((length as usize, length as usize));

        // Init for Bresenham
        let dx: i64 = (x1 - x0).abs();
        let dy: i64 = -(y1 - y0).abs();

        let mut e: i64 = dx + dy;
        let mut e2: i64;

        let (mut ty, mut tx): (i64, i64) = (y0, x0);
        let (sx, sy): (i64, i64);

        if x0 < x1 {
            sx = 1;
        } else {
            sx = -1;
        }
        
        if y0 < y1 {
            sy = 1;
        } else {
            sy = -1;
        }

        // Mask up to endpoint
        loop {
            mask[[ty as usize, tx as usize]] = 1;

            // Bresenham
            if (tx == x1) & (ty == y1) {
                break;
            }

            e2 = 2*e;

            if e2 >= dy {
                e += dy;
                tx += sx;
            }

            if e2 <= dx {
                e += dx;
                ty += sy;
            }
        }

        mask
    }

    /// Move player with `self_id` from `pos_1` to `pos_2`.
    /// Returns zeros on success and last valid point on collision.
    #[pyfn(m, "move_player")]
    #[text_signature = "(self_id, pos_1, pos_2, height_map, player_id_map, /)"]
    fn move_player_py<'py>(
        py: Python<'py>,
        self_id: i16,
        pos_1: PyReadonlyArray1<f64>,
        pos_2: PyReadonlyArray1<f64>,
        height_map: PyReadonlyArray2<u8>,
        player_id_map: PyReadonlyArray2<i16>
    ) -> &'py PyArray1<f64> {

        let pos_1 = pos_1.as_array();
        let pos_2 = pos_2.as_array();
        let height_map = height_map.as_array();
        let player_id_map = player_id_map.as_array();

        move_player(self_id, pos_1, pos_2, height_map, player_id_map).into_pyarray(py)
    }

    /// Move object with `self_id` from `pos_1` to `pos_2`.
    /// Returns zeros on success and last valid point on collision.
    #[pyfn(m, "move_object")]
    #[text_signature = "(self_id, pos_1, pos_2, wall_map, object_map, /)"]
    fn move_object_py<'py>(
        py: Python<'py>,
        self_id: i16,
        pos_1: PyReadonlyArray1<f64>,
        pos_2: PyReadonlyArray1<f64>,
        wall_map: PyReadonlyArray2<u8>,
        object_map: PyReadonlyArray2<i16>
    ) -> &'py PyArray1<f64> {

        let pos_1 = pos_1.as_array();
        let pos_2 = pos_2.as_array();
        let wall_map = wall_map.as_array();
        let object_map = object_map.as_array();

        move_object(self_id, pos_1, pos_2, wall_map, object_map).into_pyarray(py)
    }

    /// Trace shot cast by player with `self_id` from `pos_1` in direction of `pos_2`.
    /// Returns the point of a hit or zeros on reaching end of range.
    #[pyfn(m, "trace_shot")]
    #[text_signature = "(self_id, pos_1, pos_2, height_map, player_id_map, /)"]
    fn trace_shot_py<'py>(
        py: Python<'py>,
        self_id: i16,
        pos_1: PyReadonlyArray1<f64>,
        pos_2: PyReadonlyArray1<f64>,
        height_map: PyReadonlyArray2<u8>,
        player_id_map: PyReadonlyArray2<i16>
    ) -> &'py PyArray1<f64> {

        let pos_1 = pos_1.as_array();
        let pos_2 = pos_2.as_array();
        let height_map = height_map.as_array();
        let player_id_map = player_id_map.as_array();

        trace_shot(self_id, pos_1, pos_2, height_map, player_id_map).into_pyarray(py)
    }

    /// Trace line of sight of player with `self_id` from `pos_1` to `pos_2`.
    /// Returns zeros on success and last valid point on occlusion.
    #[pyfn(m, "trace_sight")]
    #[text_signature = "(self_id, pos_1, pos_2, height_map, player_id_map, zone_map, /)"]
    fn trace_sight_py<'py>(
        py: Python<'py>,
        self_id: i16,
        pos_1: PyReadonlyArray1<f64>,
        pos_2: PyReadonlyArray1<f64>,
        height_map: PyReadonlyArray2<u8>,
        player_id_map: PyReadonlyArray2<i16>,
        zone_map: PyReadonlyArray2<u8>
    ) -> &'py PyArray1<f64> {

        let pos_1 = pos_1.as_array();
        let pos_2 = pos_2.as_array();
        let height_map = height_map.as_array();
        let player_id_map = player_id_map.as_array();
        let zone_map = zone_map.as_array();

        trace_sight(self_id, pos_1, pos_2, height_map, player_id_map, zone_map).into_pyarray(py)
    }

    /// Cast rays from a preset starting point towards all given endpoints, masking unoccluded points.
    /// This is done separately for left and right parts of the view.
    #[pyfn(m, "mask_visible")]
    #[text_signature = "(self_id, height_map, player_id_map, occlusion_map, left_ends_y, left_ends_x, right_ends_y, right_ends_x, /)"]
    fn mask_visible_py<'py>(
        py: Python<'py>,
        self_id: i16,
        height_map: PyReadonlyArray2<u8>,
        player_id_map: PyReadonlyArray2<i16>,
        zone_map: PyReadonlyArray2<u8>,
        left_ends_y: PyReadonlyArray1<i64>,
        left_ends_x: PyReadonlyArray1<i64>,
        right_ends_y: PyReadonlyArray1<i64>,
        right_ends_x: PyReadonlyArray1<i64>
    ) -> &'py PyArray2<u8> {

        let height_map = height_map.as_array();
        let player_id_map = player_id_map.as_array();
        let zone_map = zone_map.as_array();
        let left_ends_y = left_ends_y.as_array();
        let left_ends_x = left_ends_x.as_array();
        let right_ends_y = right_ends_y.as_array();
        let right_ends_x = right_ends_x.as_array();

        mask_visible(self_id, height_map, player_id_map, zone_map, left_ends_y, left_ends_x, right_ends_y, right_ends_x).into_pyarray(py)
    }

    /// Cast ray from `pos_1`, the centre of a square with given side `length`, towards an endpoint `pos_2`,
    /// masking points that are traversed along the way.
    #[pyfn(m, "mask_ray")]
    #[text_signature = "(diameter, pos_1, pos_2, /)"]
    fn mask_ray_py<'py>(
        py: Python<'py>,
        length: i16,
        pos_1: PyReadonlyArray1<f64>,
        pos_2: PyReadonlyArray1<f64>,
    ) -> &'py PyArray2<u8> {

        let pos_1 = pos_1.as_array();
        let pos_2 = pos_2.as_array();

        mask_ray(length, pos_1, pos_2).into_pyarray(py)
    }

    Ok(())
}
