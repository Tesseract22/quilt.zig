// fn fft(a: *CfArr, invert: bool, allocator: std.mem.Allocator) void {
//     const n = a.items.len;
//     if (n == 1) return;
//     const half = n/2;
//     var a0 = CfArr.initCapacity(allocator, half) catch unreachable;
//     var a1 = CfArr.initCapacity(allocator, half) catch unreachable;
//     a0.resize(half) catch unreachable;
//     a1.resize(half) catch unreachable;
//     defer a0.deinit();
//     defer a1.deinit();
//     var i: usize = 0;
//     while (2 * i < 0) : (i += 1) {
//         a0[i] = a[2*i];
//         a1[i] = a[2*i+1];
//     }
//     fft(&a0, invert, allocator);
//     fft(&a1, invert, allocator);
//     const inv: f32 = if (invert) -1.0 else 1.0;
//     const angle: f32 = 2 * std.math.pi / @as(f32, @floatFromInt(n)) * inv;
//     var w = cf.init(1, 1);
//     const wn = cf.init(@cos(angle), @sin(angle));
//     i = 0;
//     while (2 * i < 0) : (i += 1) {
//         a[i] = w.mul(a1[i]).add(a0[i]);
//         a1[i + n/2] = w.mul(a1[i]).sub(a0[i]);
//         if (invert) {
//             a[i] = a[i].div(@splat(2));
//             a[i + n/2] = a[i + n/2].div(@splat(2));
//         }
//         w = w.mul(wn);
//     }
    
// }
// fn fft2dPadding(image: ImageF32, other_shape: @TypeOf(image.shape), invert: bool, allocator: std.mem.Allocator) ImageF32 {
//     const half_h = (other_shape[0]+1)/2;
//     const half_w = (other_shape[1]+1)/2;
//     const old_h, const old_w = image.shape;
//     const new_shape: @TypeOf(image.shape) = .{ old_h + other_shape[0], old_w + other_shape[1]};
//     const new_image = ImageF32.init(new_shape, allocator) catch unreachable;
//     defer allocator.free(new_image.data);
//     for (0..new_shape[0]) |y| {
//         for (0..new_shape[1]) |x| {
//             const val: Vec4 = 
//                 if (y >= half_h and y - half_h < old_h and x >= half_w and x - half_w < old_w) 
//                 image.atUnchecked(.{y - half_h, x - half_w}) 
//                 else 
//                 @splat(0);
//             new_image.setUnchecked(.{y, x}, val);
//         }
//     }
//     return fft2D(new_image, invert, allocator);
// }

// fn fft2D(image: ImageF32, invert: bool, allocator: std.mem.Allocator) ImageF32 {
//     var channels = splitRGBA(image, allocator);
//     for (&channels) |*c| {
//         fft2dOnChannel(c, invert, allocator);
//     }
//     defer for (0..4) |c| {
//         allocator.free(channels[c].data);
//     };
//     return combineRGBA(channels, allocator);
// }
// fn fft2dOnChannel(channel: *ChannelF32, invert: bool, allocator: std.mem.Allocator) void {
//     const h, const w = channel.shape;
//     var tmp_row = CfArr.initCapacity(allocator, w) catch unreachable;
//     var tmp_col = CfArr.initCapacity(allocator, h) catch unreachable;
//     tmp_row.resize(w) catch unreachable;
//     tmp_col.resize(h) catch unreachable;
//     defer tmp_row.deinit();
//     defer tmp_col.deinit();
//     for (0..h) |y| {
//         for (0..w) |x| {
//             tmp_row.items[x].re = channel.atUnchecked(.{y, x});
//             tmp_row.items[x].im = 0;
//         }
//         fft(&tmp_row, invert, allocator);
//         for (0..w) |x| {
//             channel.setUnchecked(.{y, x}, tmp_row.items[x].re);
//         }
//     }
//     for (0..w) |x| {
//         for (0..h) |y| {
//             tmp_col.items[y].re = channel.atUnchecked(.{y, x});
//             tmp_col.items[y].im = 0;
//         }
//         fft(&tmp_col, invert, allocator);
//         for (0..h) |y| {
//             channel.setUnchecked(.{y, x}, tmp_col.items[y].re);
//         }
//     }
// }
// fn fft2DOnChannelCopy(channel: ChannelF32, allocator: std.mem.Allocator) ChannelF32 {
//     const h, const w = channel.shape;
//     var tmp_row = CfArr.initCapacity(allocator, w) catch unreachable;
//     var tmp_col = CfArr.initCapacity(allocator, h) catch unreachable;
//     defer tmp_row.deinit() catch unreachable;
//     defer tmp_col.deinit() catch unreachable;
//     tmp_row.resize(w);
//     tmp_col.resize(h);
//     const out = channel.clone(allocator) catch unreachable;
//     for (0..h) |y| {
//         for (0..w) |x| {
//             tmp_row.items[x].re = out.atUnchecked(.{y, x});
//             tmp_row.items[x].im = 0;
//         }
//         fft(&tmp_row, false, allocator);
//         for (0..w) |x| {
//             out.set(.{y, x}, tmp_row.items[x].re);
//         }
//     }
//     for (0..w) |x| {
//         for (0..h) |y| {
//             tmp_col.items[y].re = out.atUnchecked(.{y, x});
//             tmp_col.items[y].im = 0;
//         }
//         fft(&tmp_col, false, allocator);
//         for (0..h) |y| {
//             out.set(.{y, x}, tmp_col.items[y].re);
//         }   
//     }
//     return out;
// }
// fn dotProduct(a: ImageF32, b: ImageF32, allocator: std.mem.Allocator) ImageF32 {
//     std.debug.assert(a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1]);
//     const res = ImageF32.init(a.shape, allocator) catch unreachable;
//     for (a.data, b.data, 0..) |ap, bp, i| {
//         res.data[i] = ap * bp;
//     }
//     return res;
// }
// fn filter2DNaive(src: ImageF32, filter: ImageF32, allocator: std.mem.Allocator) !ImageF32 {
//     std.debug.assert(filter.shape[0] % 2 == 1 and filter.shape[1] % 2 == 1);
//     const h = src.shape[0];
//     const w = src.shape[1];
//     const filter_h, const filter_w = filter.shape;
//     const half_h = filter_h/2;
//     const half_w = filter_w/2;
//     var res = try ImageF32.init(src.shape, allocator);
//     for (0..h) |y| {
//         for (0..w) |x| {
//             var sum: Vec4 = @splat(0);
//             for (0..filter_h) |fy| {
//                 for (0..filter_w) |fx| {
//                     const image_val = 
//                         if (y + fy >= half_h and y + fy - half_h < h and x + fx >= half_w and x + fx - half_w < w)
//                         src.atUnchecked(.{y + fy - half_h, x + fx - half_w})
//                         else 
//                         Vec4 {0, 0, 0, 0};
//                     const filter_val = filter.atUnchecked(.{fy, fx});
//                     // std.debug.print("{}\n", .{filter_val});

//                     sum += filter_val * image_val;

//                 }
//             }
//             res.setUnchecked(.{y, x}, sum);

//         }
//     }
//     return res;
// }