const std = @import("std");
const fft = @import("convultion.zig");
// const raylib = @cImport(@cInclude("raylib.h"));
const lodepng = @cImport(@cInclude("lodepng.h"));
pub fn ssdPatch(template: fft.ImageF32, image: fft.ImageF32, allocator: std.mem.Allocator) fft.FMat {
    const masked_template = template.clone(allocator) catch unreachable;
    const mask = fft.FMat.init(template.shape, allocator) catch unreachable;
    defer {
        masked_template.deinit();
        mask.deinit();
    }
    @memset(mask.data, 0);
    @memset(masked_template.data, @splat(0));
    for (0..template.shape[0]) |y| {
        for (0..template.shape[1]) |x| {
            const y_flip = template.shape[0] - 1 - y;
            const x_flip = template.shape[1] - 1 - x;
            const t = template.atUnchecked(.{y, x});
            if (t[0] == -1) {
                masked_template.setUnchecked(.{y_flip, x_flip}, @splat(0));
                mask.setUnchecked(.{y_flip, x_flip}, 0);
            } else {
                masked_template.setUnchecked(.{y_flip, x_flip}, t);
                mask.setUnchecked(.{y_flip, x_flip}, 1);
            }
        }
    }
    const res = fft.FMat.init(image.shape, allocator) catch unreachable; // TODO: arena
    // T ** 2
    var t2: @Vector(4, f32) = @splat(0);
    for (masked_template.data) |mt| {
        t2 += mt * mt;
    }

    // t2[3] = 0; // TODO: properly ignore alpha channel
    // I * T
    const i_t = image.convultImage(masked_template, .Reflect, allocator);
    defer i_t.deinit();
    // I ** 2
    const image2 = image.clone(allocator) catch unreachable; // TODO: arena
    defer image2.deinit();
    for (image2.data) |*i| {
        i.* = i.* * i.*;
    }
    
    const i2c = image2.convult(mask, .Reflect, allocator);
    defer i2c.deinit();

    
    // sum
    for (res.data, i_t.data, i2c.data) |*r, it, i2ci| {
        r.* = @reduce(.Add, t2) - 2 * @reduce(.Add, it) + @reduce(.Add, i2ci);
        r.* -= t2[3] - 2 * it[3] + i2ci[3];
    }

    return res;
    
}
const ArgSort = struct {
    val: f32,
    i: usize,
    pub fn lessThanFn(ctx: void, a: ArgSort, b: ArgSort) bool {
        _ = ctx;
        return a.val < b.val;
    }
};
pub fn chooseSample(ssd: fft.FMat, half: usize, r: usize, allocator: std.mem.Allocator) [2]usize {

    const edge_removed = fft.Array(ArgSort, 2).init(ssd.shape, allocator) catch unreachable;
    for (0..ssd.shape[0]) |y| {
        for (0..ssd.shape[1]) |x| {
            const val = 
                if (y < half or y > ssd.shape[0] - half - 1 or x < half or x > ssd.shape[1] - half - 1) std.math.inf(f32) else ssd.atUnchecked(.{y, x});
            edge_removed.setUnchecked(.{y, x}, .{.val = val, .i = edge_removed.index(.{y, x})});
        }
    }
    std.mem.sort(ArgSort, edge_removed.data, void {}, ArgSort.lessThanFn);
    return edge_removed.unIndex(edge_removed.data[r].i);
    

}
pub fn quiltSimple(sample: fft.ImageF32, out_shape: [2]usize, 
    patch_size: usize, overlap: usize, 
    tol: usize, allocator: std.mem.Allocator) fft.ImageF32 {

    std.debug.assert(out_shape[0] > patch_size and out_shape[1] > patch_size);
    std.debug.assert(patch_size > overlap);
    std.debug.assert(tol > 0);
    const half_patch = patch_size / 2;
    const out = fft.ImageF32.init(out_shape, allocator) catch unreachable;
    @memset(out.data, @splat(-1));
    const step = patch_size - overlap;
    var rand = std.rand.DefaultPrng.init(0);
    // the first patch
    const xr = rand.next() % (sample.shape[1] - patch_size);
    const yr = rand.next() % (sample.shape[0] - patch_size);
    for (0..patch_size) |y| {
        for (0..patch_size) |x| {
            out.setUnchecked(.{y, x}, sample.atUnchecked(.{yr+y, xr+x}));
        }
    }

    var i: usize = 0;
    var j: usize = step;
    while (i < out_shape[0] - patch_size): (i += step) { // y
        while (j < out_shape[1] - patch_size): (j += step) { // x
            const template = fft.ImageF32.init(.{patch_size, patch_size}, allocator) catch unreachable; // TODO
            for (0..patch_size) |y| {
                for (0..patch_size) |x| {
                    template.setUnchecked(.{y, x}, out.atUnchecked(.{y + i, x + j}));
                }
            }
            const ssd = ssdPatch(template, sample, allocator); // TODO: arena allocator
            const pos = chooseSample(ssd, half_patch, rand.next() % tol, allocator);
            std.debug.print("pos: {any} : {any}\n", .{pos, ssd.atUnchecked(pos) });
            for (0..patch_size) |y| {
                for (0..patch_size) |x| {
                    out.setUnchecked(.{y + i, x + j}, sample.atUnchecked(.{y + pos[0] - half_patch, x + pos[1] - half_patch}));
                }
            }
        }
        j = 0;
    }
    for (out.data) |*v| {
        for (0..4) |c| {
            if (v[c] == -1) v[c] = 0;
        }
    }
    return out;
}

pub fn findSeam(cost: fft.FMat, vertical: bool, allocator: std.mem.Allocator) []usize {
    // find from to bottom
    const h, const w = if (vertical) cost.shape else .{cost.shape[1], cost.shape[0]};
    const res = allocator.alloc(usize, h) catch unreachable;
    const cost_acc = cost.clone(allocator) catch unreachable;
    const cost_path = fft.Array(usize, 2).init(cost.shape, allocator) catch unreachable;
    defer cost_acc.deinit();
    defer cost_path.deinit(); 
    const inf = std.math.inf(f32);
    for (0..h) |y| {
        for (0..w) |x| {
            if (y == 0) continue;
            const prev_row = y - 1;
            const idxes  = [3]usize {if (x > 0) x - 1 else x, x, if (x < w-1) x + 1 else x};
            var min_cost = inf;
            var min_idx = idxes[0];
            for (idxes) |i| {
                const c = (if (vertical)
                    cost.at(.{prev_row, i}) else cost.at(.{i, prev_row})) catch inf;
                if (c < min_cost) {
                    min_cost = c;
                    min_idx = i;
                }
            }
            const curr: [2]usize = if (vertical) .{y, x} else .{x, y};
            cost_acc.getUnchecked(curr).* += min_cost;
            cost_path.setUnchecked(curr, min_idx);
        }
    }
    var min_idx: usize = 0;
    var min_cost = inf;
    for (0..w) |x| {
        const c = 
            if (vertical) cost_acc.atUnchecked(.{h - 1, x})
            else cost_acc.atUnchecked(.{x, h - 1});
        if (min_cost < c) {
            min_cost = c;
            min_idx = x;
        }
    }
    var row = h - 1;
    while (row > 0): (row -= 1) {
        res[row] = min_idx;
        min_idx = 
            if (vertical) cost_path.atUnchecked(.{row, min_idx})
            else cost_path.atUnchecked(.{min_idx, row});
    }
    res[row] = min_idx;
    min_idx = 
            if (vertical) cost_path.atUnchecked(.{row, min_idx})
            else cost_path.atUnchecked(.{min_idx, row});

    return res;
}


pub fn quiltSeam(sample: fft.ImageF32, out_shape: [2]usize, 
    patch_size: usize, overlap: usize, 
    tol: usize, allocator: std.mem.Allocator) fft.ImageF32 {

    std.debug.assert(out_shape[0] > patch_size and out_shape[1] > patch_size);
    std.debug.assert(patch_size > overlap);
    std.debug.assert(tol > 0);
    const half_patch = patch_size / 2;
    const out = fft.ImageF32.init(out_shape, allocator) catch unreachable;
    @memset(out.data, @splat(-1));
    const step = patch_size - overlap;
    var rand = std.rand.DefaultPrng.init(0);
    // the first patch
    const xr = rand.next() % (sample.shape[1] - patch_size);
    const yr = rand.next() % (sample.shape[0] - patch_size);
    for (0..patch_size) |y| {
        for (0..patch_size) |x| {
            out.setUnchecked(.{y, x}, sample.atUnchecked(.{yr+y, xr+x}));
        }
    }

    var i: usize = 0;
    var j: usize = step;
    while (i < out_shape[0] - patch_size): (i += step) { // y
        while (j < out_shape[1] - patch_size): (j += step) { // x
            const template = fft.ImageF32.init(.{patch_size, patch_size}, allocator) catch unreachable; // TODO
            for (0..patch_size) |y| {
                for (0..patch_size) |x| {
                    template.setUnchecked(.{y, x}, out.atUnchecked(.{y + i, x + j}));
                }
            }
            const ssd = ssdPatch(template, sample, allocator); // TODO: arena allocator
            const pos = chooseSample(ssd, half_patch, rand.next() % tol, allocator);

            std.log.debug("pos: {any} : {any}\n", .{pos, ssd.atUnchecked(pos) });


            for (0..patch_size) |y| {
                for (0..patch_size) |x| {
                    template.setUnchecked(.{y, x}, sample.atUnchecked(.{y + pos[0] - half_patch, x + pos[1] - half_patch}));
                }
            }

            if (i > 0) {
                const top_cost = fft.FMat.init(.{overlap, patch_size}, allocator) catch unreachable;
                defer top_cost.deinit();
                for (0..overlap) |y| {
                    for (0..patch_size) |x| {
                    const new = sample.atUnchecked(.{y + pos[0] - half_patch, x + pos[1] - half_patch});
                    const old = out.atUnchecked(.{y + i, x + j});
                    top_cost.setUnchecked(.{y, x}, @reduce(.Add, (new - old) * (new - old)));
                    }   
                }
                const top_path = findSeam(top_cost, false, allocator);
                defer allocator.free(top_path);
                for (0..patch_size) |x| {
                    const y_end = top_path[x];
                    for (0..y_end) |y| {
                        template.setUnchecked(.{y, x}, out.atUnchecked(.{y + i, x + j}));
                    }
                }
            }
            if (j > 0) {
                const left_cost = fft.FMat.init(.{patch_size, overlap}, allocator) catch unreachable;
                defer left_cost.deinit();
                for (0..patch_size) |y| {
                    for (0..overlap) |x| {
                        const new = sample.atUnchecked(.{y + pos[0] - half_patch, x + pos[1] - half_patch});
                        const old = out.atUnchecked(.{y + i, x + j});
                        left_cost.setUnchecked(.{y, x}, @reduce(.Add,(new - old) * (new - old)));
                    }
                }
                const left_path = findSeam(left_cost, true, allocator);
                defer allocator.free(left_path);
                for (0..patch_size) |y| {
                    const x_end = left_path[y];
                    for (0..x_end) |x| {
                        template.setUnchecked(.{y, x}, out.atUnchecked(.{y + i, x + j}));
                    }
                }

            }
            for (0..patch_size) |y| {
                for (0..patch_size) |x| {
                    out.setUnchecked(.{y + i, x + j}, template.atUnchecked(.{y, x}));
                }
            }
        }
        j = 0;
    }
    for (out.data) |*v| {
        for (0..4) |c| {
            if (v[c] == -1) v[c] = 0;
        }
    }
    return out;
}

pub fn quiltTransfer(sample: fft.ImageF32, target: fft.ImageF32, a: f32,
    patch_size: usize, overlap: usize, 
    tol: usize, allocator: std.mem.Allocator) fft.ImageF32 {

    const out_shape = target.shape;
    const half_patch = patch_size / 2;
    const step = patch_size - overlap;

    std.debug.assert(out_shape[0] > patch_size and out_shape[1] > patch_size);
    std.debug.assert(patch_size > overlap);
    std.debug.assert(tol > 0);
    std.debug.assert(a >= 0 and a <= 1);

    const out = fft.ImageF32.init(out_shape, allocator) catch unreachable;
    @memset(out.data, @splat(-1));
    const grey_sample = sample.toGreyScale();
    const grey_target = sample.toGreyScale();
    defer grey_sample.deinit();
    defer grey_target.deinit();

    var rand = std.rand.DefaultPrng.init(0);
    // the first patch
    const xr = rand.next() % (sample.shape[1] - patch_size);
    const yr = rand.next() % (sample.shape[0] - patch_size);
    for (0..patch_size) |y| {
        for (0..patch_size) |x| {
            out.setUnchecked(.{y, x}, sample.atUnchecked(.{yr+y, xr+x}));
        }
    }

    var i: usize = 0;
    var j: usize = step;
    while (i < out_shape[0] - patch_size): (i += step) { // y
        while (j < out_shape[1] - patch_size): (j += step) { // x
            const template = fft.ImageF32.init(.{patch_size, patch_size}, allocator) catch unreachable; // TODO
            const target_template = template.clone(allocator) catch unreachable;
            defer template.deinit();
            defer target_template.deinit();
            for (0..patch_size) |y| {
                for (0..patch_size) |x| {
                    template.setUnchecked(.{y, x}, out.atUnchecked(.{y + i, x + j}));
                    target_template.setUnchecked(.{y, x}, grey_target(.{y + i, x + j}));
                }
            }
            const ssd = ssdPatch(template, sample, allocator); // TODO: arena allocator
            const ssd_target = ssdPatch(target_template, grey_sample, allocator);
            for (ssd.data, ssd_target.data) |*s, st| {
                s.* = s.* * a + st * (1 - a);
            }
            const pos = chooseSample(ssd, half_patch, rand.next() % tol, allocator);

            std.log.debug("pos: {any} : {any}\n", .{pos, ssd.atUnchecked(pos) });


            for (0..patch_size) |y| {
                for (0..patch_size) |x| {
                    template.setUnchecked(.{y, x}, sample.atUnchecked(.{y + pos[0] - half_patch, x + pos[1] - half_patch}));
                }
            }

            if (i > 0) {
                const top_cost = fft.FMat.init(.{overlap, patch_size}, allocator) catch unreachable;
                defer top_cost.deinit();
                for (0..overlap) |y| {
                    for (0..patch_size) |x| {
                    const new = sample.atUnchecked(.{y + pos[0] - half_patch, x + pos[1] - half_patch});
                    const old = out.atUnchecked(.{y + i, x + j});
                    top_cost.setUnchecked(.{y, x}, @reduce(.Add, (new - old) * (new - old)));
                    }   
                }
                const top_path = findSeam(top_cost, false, allocator);
                defer allocator.free(top_path);
                for (0..patch_size) |x| {
                    const y_end = top_path[x];
                    for (0..y_end) |y| {
                        template.setUnchecked(.{y, x}, out.atUnchecked(.{y + i, x + j}));
                    }
                }
            }
            if (j > 0) {
                const left_cost = fft.FMat.init(.{patch_size, overlap}, allocator) catch unreachable;
                defer left_cost.deinit();
                for (0..patch_size) |y| {
                    for (0..overlap) |x| {
                        const new = sample.atUnchecked(.{y + pos[0] - half_patch, x + pos[1] - half_patch});
                        const old = out.atUnchecked(.{y + i, x + j});
                        left_cost.setUnchecked(.{y, x}, @reduce(.Add,(new - old) * (new - old)));
                    }
                }
                const left_path = findSeam(left_cost, true, allocator);
                defer allocator.free(left_path);
                for (0..patch_size) |y| {
                    const x_end = left_path[y];
                    for (0..x_end) |x| {
                        template.setUnchecked(.{y, x}, out.atUnchecked(.{y + i, x + j}));
                    }
                }

            }
            for (0..patch_size) |y| {
                for (0..patch_size) |x| {
                    out.setUnchecked(.{y + i, x + j}, template.atUnchecked(.{y, x}));
                }
            }
        }
        j = 0;
    }
    for (out.data) |*v| {
        for (0..4) |c| {
            if (v[c] == -1) v[c] = 0;
        }
    }
    return out;
}
pub fn main() !void {
    const allocator = std.heap.c_allocator;
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    const self = args.next().?;
    const in_path = args.next() orelse {
        std.debug.print("No Input file provided.\n{s} <in_path> <out_path>\n", .{self});
        unreachable;
    };
    const out_path = args.next() orelse blk: {
        std.debug.print("No Output path provided, using default \"quilt-out.png\"\n", .{});
        break :blk "quilt-out.png";
    };

    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    // image data
    var out: [*c]u8 = undefined;
    var w: c_uint = 0;
    var h: c_uint = 0;


    const decode_Err = lodepng.lodepng_decode32_file(@ptrCast(&out), @ptrCast(&w), @ptrCast(&h), in_path);
    defer std.c.free(out);
    std.log.debug("Error: {s}", .{lodepng.lodepng_error_text(decode_Err)});
    std.log.debug("w: {}, h: {}\n", .{w, h});
    const image = fft.ImageU8.initWithRaw(.{h, w, 4}, out[0..h*w*4]) catch unreachable;
    const image_f32 = image.ImageU8ToF32(allocator);
    defer image_f32.deinit();
    const out_size = [2]usize {512, 512};
    var t = std.time.milliTimestamp();
    const quilt_res = quiltTransfer(image_f32, out_size, 25, 13, 5, allocator);
    t = std.time.milliTimestamp() - t;
    try stdout.print("quilting took {} ms", .{t});
    defer quilt_res.deinit();
    const res = quilt_res.ImageF32ToU8(allocator);
    defer res.deinit();
    const encode_err = lodepng.lodepng_encode32_file(out_path, @ptrCast(res.data.ptr), @intCast(out_size[1]), @intCast(out_size[0]));
    std.log.debug("Error: {s}", .{lodepng.lodepng_error_text(encode_err)});
}

test "index" {
    const arr = try fft.FMat.init(.{10, 25}, std.testing.allocator);
    defer arr.deinit();
    try std.testing.expect(std.mem.eql(usize, &.{3, 7}, &arr.unIndex(arr.index(.{3, 7}))));
}

fn filter2DNaive(src: fft.ImageF32, filter: fft.FMat, allocator: std.mem.Allocator) fft.ImageF32 {
    std.debug.assert(filter.shape[0] % 2 == 1 and filter.shape[1] % 2 == 1);
    const h = src.shape[0];
    const w = src.shape[1];
    const filter_h, const filter_w = filter.shape;
    const half_h = filter_h/2;
    const half_w = filter_w/2;
    var res = fft.ImageF32.init(src.shape, allocator) catch unreachable;
    for (0..h) |y| {
        for (0..w) |x| {
            var sum: @Vector(4, f32) = @splat(0);
            for (0..filter_h) |fy| {
                for (0..filter_w) |fx| {
                    const image_val = 
                        if (y + fy >= half_h and y + fy - half_h < h and x + fx >= half_w and x + fx - half_w < w)
                        src.atUnchecked(.{y + fy - half_h, x + fx - half_w})
                        else 
                        @Vector(4, f32) {0, 0, 0, 0};
                    const filter_val = filter.atUnchecked(.{fy, fx});
                    // std.debug.print("{}\n", .{filter_val});

                    sum += image_val * @as(@Vector(4, f32), @splat(filter_val));

                }
            }
            res.setUnchecked(.{y, x}, sum);

        }
    }
    return res;
}
test "fft" {
    
}
// test "convult" {
//     const allocator = std.testing.allocator;
//     var rand = std.rand.DefaultPrng.init(0);
//     const random = rand.random();
//     const im = try fft.ImageF32.init(.{10, 10}, allocator);
//     defer im.deinit();
//     for (im.data) |*d| {
//         for (0..4) |c| {
//             d[c] = random.float(f32);
//         }
//     }
//     const id_kernal = try fft.FMat.init(.{5, 5}, allocator);
//     defer id_kernal.deinit();
//     @memset(id_kernal.data, 0);
//     id_kernal.setUnchecked(.{2,2}, 1);
//     const imc2 = filter2DNaive(im, id_kernal, allocator);
//     defer imc2.deinit();
//     for (imc2.data, im.data, 0..) |c, i, index| {
//         std.debug.print("{} {} == {}\n", .{index, c, i});
//         try std.testing.expect(@reduce(.And, c == i));
//     }
//     const imc = im.convult(id_kernal, allocator);
//     defer imc.deinit();
//     for (imc.data, im.data, 0..) |c, i, index| {
//         std.debug.print("{} {} == {}", .{index, c, i});
//         try std.testing.expect(@reduce(.And, c == i));
//     }

// }
// test "reflect" {
//     const allocator = std.testing.allocator;
//     const im = try fft.ImageF32.init(.{5, 5}, allocator);
//     defer im.deinit();
//     for (0..5) |y| {
//         for (0..5) |x| {
//             im.setUnchecked(.{y, x}, @splat(@floatFromInt(y + x)));
//         } 
//     }
//     const cs = im.splitRGBAToShape(.{15, 15}, .Reflect, allocator);
//     defer for (0..4) |c| cs[c].deinit();
//     for (0..15) |y| {
//         for (0..15) |x| {
//             std.debug.print("{}, ", .{cs[0].atUnchecked(.{y ,x})});
//         }
//         std.debug.print("\n", .{});
//     }
// }

test "convult" {
    const allocator = std.testing.allocator;
    const im = try fft.ImageF32.init(.{10, 10}, allocator);
    defer im.deinit();
    for (0..10) |y| {
        for (0..10) |x| {
            im.setUnchecked(.{y, x}, @splat(@floatFromInt(x + y)));
        }
    }

    const k = try fft.ImageF32.init(.{5,5}, allocator);
    defer k.deinit();
    @memset(k.data, @splat(0));
    for (2..3) |x| {
        k.setUnchecked(.{x, 2}, @splat(1));
        // k.setUnchecked(.{2, x}, 1);
    }
    // const c1 = filter2DNaive(im, k, allocator);
    // defer c1.deinit();

    const c2 = im.convultImage(k, .Reflect, allocator);
    defer c2.deinit();
    std.debug.print("\n", .{});
    for (0..10) |y| {
        for (0..10) |x| {
            std.debug.print("{d:.3}, ", .{c2.atUnchecked(.{y, x})[0]});
        }
        std.debug.print("\n", .{});
    }
}








