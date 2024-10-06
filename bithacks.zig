// Zig Bit Twiddling Hacks
// https://github.com/cryptocode/bithacks
//
// Like the original snippets, this file is released under public domain:
//
// The code and descriptions are distributed in the hope that they will
// be useful, but WITHOUT ANY WARRANTY and without even the implied
// warranty of merchantability or fitness for a particular purpose.

const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const assert = std.debug.assert;

/// Asserts at compile time that `T` is an integer, returns `T`
pub fn requireInt(comptime T: type) type {
    comptime assert(@typeInfo(T) == .int);
    return T;
}

/// Asserts at compile time that `T` is a nsigned integer, returns `T`
pub fn requireSignedInt(comptime T: type) type {
    _ = requireInt(T);
    comptime assert(@typeInfo(T).int.signedness == .signed);
    return T;
}

/// Asserts at compile time that `T` is an unsigned integer, returns `T`
pub fn requireUnsignedInt(comptime T: type) type {
    _ = requireInt(T);
    comptime assert(@typeInfo(T).int.signedness == .unsigned);
    return T;
}

/// Compute the sign of an integer
/// Returns true if the sign bit of `val` is set, otherwise false
/// https://github.com/cryptocode/bithacks#CopyIntegerSign
pub fn isSignBitSet(val: anytype) bool {
    const T = requireSignedInt(@TypeOf(val));
    return -(@as(T, @intCast(@intFromBool(val < 0)))) == -1;
}

test "Compute the sign of an integer" {
    const cases = [5]i32{ std.math.minInt(i32), -1, 0, 1, std.math.maxInt(i32) };
    const expected = [5]bool{ true, true, false, false, false };
    for (cases, 0..) |num, i| {
        try expect(isSignBitSet(num) == expected[i]);
    }
}

/// Detect if two integers have opposite signs
/// Returns true if the `first` and `second` signed integers have opposite signs
/// https://github.com/cryptocode/bithacks#detect-if-two-integers-have-opposite-signs
pub fn isOppositeSign(first: anytype, second: @TypeOf(first)) bool {
    _ = requireSignedInt(@TypeOf(first));
    return (first ^ second) < 0;
}

test "Detect if two integers have opposite signs" {
    try expect(isOppositeSign(@as(i32, -1), @as(i32, 1)));
    try expect(!isOppositeSign(@as(i32, 1), @as(i32, 1)));
}

/// Compute the integer absolute value (abs) without branching
/// https://github.com/cryptocode/bithacks#compute-the-integer-absolute-value-abs-without-branching
pub fn absFast(val: anytype) @TypeOf(val) {
    const T = requireSignedInt(@TypeOf(val));
    const bits = @typeInfo(T).int.bits;

    const mask: T = val >> (bits - 1);
    return (val + mask) ^ mask;
}

test "Compute the integer absolute value (abs) without branching" {
    const cases = [5]i32{ std.math.minInt(i32) + 1, -1, 0, 1, std.math.maxInt(i32) };
    const expected = [5]i32{ std.math.maxInt(i32), 1, 0, 1, std.math.maxInt(i32) };
    for (cases, 0..) |num, i| {
        try expect(absFast(num) == expected[i]);
    }
}

/// Find the minimum of two integers without branching
/// https://github.com/cryptocode/bithacks#compute-the-minimum-min-or-maximum-max-of-two-integers-without-branching
pub fn minFast(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    _ = requireSignedInt(@TypeOf(x));
    return y ^ ((x ^ y) & -@as(@TypeOf(x), @intCast(@intFromBool((x < y)))));
}

/// Find the maximum of two signed integers without branching
/// https://github.com/cryptocode/bithacks#compute-the-minimum-min-or-maximum-max-of-two-integers-without-branching
pub fn maxFast(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    _ = requireSignedInt(@TypeOf(x));
    return x ^ ((x ^ y) & -@as(@TypeOf(x), @intCast(@intFromBool((x < y)))));
}

test "Compute the minimum (min) or maximum (max) of two integers without branching" {
    const x: i32 = 19;
    const y: i32 = -13;
    try expect(minFast(x, y) == y);
    try expect(maxFast(x, y) == x);
}

/// Determining if an integer is a power of 2
/// Generic function that checks if the input integer is a power of 2. If the input
/// is a signed integer, the generated function will include a call to absFast()
/// https://github.com/cryptocode/bithacks#determining-if-an-integer-is-a-power-of-2
pub fn isPowerOf2(val: anytype) bool {
    const T = @TypeOf(val);
    const abs = if (@typeInfo(T) == .int and @typeInfo(T).int.signedness == .signed) absFast(val) else val;
    return abs != 0 and (abs & (abs - 1)) == 0;
}

test "Determining if an integer is a power of 2" {
    try expect(isPowerOf2(@as(i32, -64)));
    try expect(!isPowerOf2(@as(i32, -63)));
    try expect(isPowerOf2(@as(u32, 64)));
    try expect(!isPowerOf2(@as(u32, 63)));
    try expect(!isPowerOf2(@as(u32, 0)));
}

/// Sign extending from a constant bit-width
/// Input `val` is an unsigned n-bit numbers that is reinterpreted as a signed integer which
/// is then signed-extended to the `target` type and returned.
/// https://github.com/cryptocode/bithacks#sign-extending-from-a-constant-bit-width
pub fn signExtendFixed(comptime target: type, val: anytype) target {
    const T = requireUnsignedInt(@TypeOf(val));
    const SignedType = std.meta.Int(.signed, @typeInfo(T).int.bits);
    return @as(SignedType, @bitCast(val));
}

test "Sign extending from a constant bit-width2" {
    // Input is -3 in 4-bit two's complement representation, which we sign extend to an i16
    try expectEqual(signExtendFixed(i16, @as(u4, 0b1101)), -3);
    try expectEqual(signExtendFixed(i16, @as(u5, 0b10000)), -16);
}

/// Sign extending from a variable bit-width
/// The `val` argument is an integer with size >= `bits`, but only `bits` number of bits actually
/// represents the number to be sign-extended to the `target` type.
/// https://github.com/cryptocode/bithacks#sign-extending-from-a-variable-bit-width
pub fn signExtendVariable(comptime target: type, comptime bits: usize, val: anytype) target {
    return @as(target, @as(std.meta.Int(.signed, bits), @truncate(val)));
}

test "Sign extending from a variable bit-width" {
    // Input is 0b10110110, but we only care about the lower 3 bits which we sign extend into an i16
    const res = signExtendVariable(i16, 3, @as(i8, @bitCast(@as(u8, 0b10110110))));
    try expectEqual(res, -2);
}

/// Conditionally set or clear bits without branching
/// https://github.com/cryptocode/bithacks#conditionally-set-or-clear-bits-without-branching
pub fn setOrClearBits(set: bool, mask: anytype, val: anytype) @TypeOf(val) {
    _ = requireInt(@TypeOf(mask));
    const T = requireInt(@TypeOf(val));

    return (val & ~mask) | (-%@as(T, @intFromBool(set)) & mask);
}

test "Conditionally set or clear bits without branching" {
    const mask: u8 = 0b10110010;
    const bits: u8 = 0b01000011;

    var res = setOrClearBits(true, mask, bits);
    try expect(res == 0b11110011);

    res = setOrClearBits(false, mask, bits);
    try expectEqual(res, 0b01000001);
}

/// Conditionally negate a value without branching
/// https://github.com/cryptocode/bithacks#conditionally-negate-a-value-without-branching
pub fn negateIf(negate: bool, val: anytype) @TypeOf(val) {
    const T = requireSignedInt(@TypeOf(val));
    const negate_as_int = @as(T, @intFromBool(negate));
    return (val ^ -negate_as_int) + negate_as_int;
}

test "Conditionally negate a value without branching" {
    try expectEqual(negateIf(true, @as(i32, 50)), -50);
    try expectEqual(negateIf(false, @as(i32, 50)), 50);
}

/// Merge bits from two values according to a mask"
/// https://github.com/cryptocode/bithacks#merge-bits-from-two-values-according-to-a-mask
pub fn mergeBits(first: anytype, second: @TypeOf(first), mask: @TypeOf(first)) @TypeOf(first) {
    _ = requireUnsignedInt(@TypeOf(first));
    return first ^ ((first ^ second) & mask);
}

test "Merge bits from two values according to a mask" {
    const a: u8 = 0b10110010;
    const b: u8 = 0b00001101;

    // 1 = which bits to pick from a
    // 0 = which bits to pick from b
    const m: u8 = 0b00001111;

    try expectEqual(mergeBits(a, b, m), 0b10111101);
}

/// Counting bits set (naive way)
/// https://github.com/cryptocode/bithacks#counting-bits-set-naive-way
pub fn countBitsSetNaive(val: anytype) usize {
    const T = requireInt(@TypeOf(val));

    var v = val;
    var bits_set: T = 0;
    while (v != 0) : (v >>= 1) {
        bits_set +%= v & 1;
    }
    return @as(usize, @intCast(bits_set));
}

test "Counting bits set (naive way)" {
    try expectEqual(countBitsSetNaive(@as(u8, 0b0)), 0);
    try expectEqual(countBitsSetNaive(@as(u8, 0b11100011)), 5);
    try expectEqual(countBitsSetNaive(@as(u8, 0b11111111)), 8);
    try expectEqual(countBitsSetNaive(@as(i8, 0b1111111)), 7);
    try expectEqual(countBitsSetNaive(@as(u32, 0xffffffff)), 32);
    try expectEqual(countBitsSetNaive(@as(u64, 0xffffffffffffffff)), 64);
}

/// Counting bits set by lookup table
/// https://github.com/cryptocode/bithacks#counting-bits-set-by-lookup-table
pub fn countBitsByLookupTable(val: u32) usize {
    // Generate the lookup table at compile time
    const bitSetTable = comptime val: {
        var table: [256]u8 = undefined;
        table[0] = 0;
        var i: usize = 0;
        while (i < 256) : (i += 1) {
            table[i] = (i & 1) + table[i / 2];
        }

        break :val table;
    };

    return bitSetTable[val & 0xff] +
        bitSetTable[(val >> 8) & 0xff] +
        bitSetTable[(val >> 16) & 0xff] +
        bitSetTable[val >> 24];
}

test "Counting bits set by lookup table" {
    try expectEqual(countBitsByLookupTable(0b0), 0);
    try expectEqual(countBitsByLookupTable(0b11100011), 5);
    try expectEqual(countBitsByLookupTable(0b1111111), 7);
    try expectEqual(countBitsByLookupTable(0b11111111), 8);
    try expectEqual(countBitsByLookupTable(0xffffffff), 32);
}

/// Counting bits set, Brian Kernighan's way
/// https://github.com/cryptocode/bithacks#counting-bits-set-brian-kernighans-way
pub fn countBitsSetKernighan(val: anytype) usize {
    _ = requireInt(@TypeOf(val));
    var v = val;
    var bits_set: usize = 0;
    while (v != 0) : (bits_set += 1) {
        v &= v - 1;
    }
    return @as(usize, @truncate(bits_set));
}

test "Counting bits set, Brian Kernighan's way" {
    try expectEqual(countBitsSetKernighan(@as(u8, 0b0)), 0);
    try expectEqual(countBitsSetKernighan(@as(u8, 0b11100011)), 5);
    try expectEqual(countBitsSetKernighan(@as(u8, 0b11111111)), 8);
    try expectEqual(countBitsSetKernighan(@as(i8, 0b1111111)), 7);
    try expectEqual(countBitsSetKernighan(@as(u32, 0xffffffff)), 32);
    try expectEqual(countBitsSetKernighan(@as(u64, 0xffffffffffffffff)), 64);
}

/// Counting bits set in 14, 24, or 32-bit words using 64-bit instructions
/// https://github.com/cryptocode/bithacks#counting-bits-set-in-14-24-or-32-bit-words-using-64-bit-instructions
pub fn countBitsSetModulus(val: anytype) usize {
    const T = requireInt(@TypeOf(val));
    const bits_set: u64 = switch (@typeInfo(T).int.bits) {
        14 => (val * @as(u64, 0x200040008001) & @as(u64, 0x111111111111111)) % 0xf,
        24 => res: {
            var c: u64 = ((@as(u64, @intCast(val)) & 0xfff) * @as(u64, 0x1001001001001) & @as(u64, 0x84210842108421)) % 0x1f;
            c += (((@as(u64, @intCast(val)) & 0xfff000) >> 12) * @as(u64, 0x1001001001001) & @as(u64, 0x84210842108421)) % 0x1f;
            break :res c;
        },
        32 => res: {
            var c: u64 = ((val & 0xfff) * @as(u64, 0x1001001001001) & @as(u64, 0x84210842108421)) % 0x1f;
            c += (((val & 0xfff000) >> 12) * @as(u64, 0x1001001001001) & @as(u64, 0x84210842108421)) % 0x1f;
            c += ((val >> 24) * @as(u64, 0x1001001001001) & @as(u64, 0x84210842108421)) % 0x1f;
            break :res c;
        },
        else => @panic("Invalid integer size"),
    };

    return @as(usize, @truncate(bits_set));
}

test "Counting bits set in 14, 24, or 32-bit words using 64-bit instructions" {
    try expectEqual(countBitsSetModulus(@as(u14, 0b11111111111110)), 13);
    try expectEqual(countBitsSetModulus(@as(u14, 0b11111111111111)), 14);
    try expectEqual(countBitsSetModulus(@as(u24, 0b111111111111111111111110)), 23);
    try expectEqual(countBitsSetModulus(@as(u24, 0b111111111111111111111111)), 24);
    try expectEqual(countBitsSetModulus(@as(u32, 0b0)), 0);
    try expectEqual(countBitsSetModulus(@as(u32, 0b11100011)), 5);
    try expectEqual(countBitsSetModulus(@as(u32, 0b11111111)), 8);
    try expectEqual(countBitsSetModulus(@as(u32, 0xfffffffe)), 31);
    try expectEqual(countBitsSetModulus(@as(u32, 0xffffffff)), 32);
}

/// Counting bits set, in parallel
/// https://github.com/cryptocode/bithacks#counting-bits-set-in-parallel
pub fn countBitsSetParallel(val: anytype) @TypeOf(val) {
    const T = requireUnsignedInt(@TypeOf(val));
    var v = val;
    var bits_set: T = 0;
    const ones = ~@as(T, 0);

    switch (@typeInfo(T).int.bits) {
        // Method optimized for 32 bit integers
        32 => {
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            bits_set = ((v + (v >> 4) & 0xF0F0F0F) *% 0x1010101) >> 24;
        },
        // Generalized version for integers up to 128 bits in width
        else => |bits| {
            v = v - ((v >> 1) & @as(T, ones / 3));
            v = (v & @as(T, ones / 15 * 3)) + ((v >> 2) & @as(T, ones / 15 * 3));
            v = (v + (v >> 4)) & @as(T, ones / 255 * 15);
            bits_set = @as(T, (v *% (@as(T, ones / 255))) >> (bits / 8 - 1) * 8);
        },
    }

    return bits_set;
}

test "Counting bits set, in parallel" {
    try expectEqual(countBitsSetParallel(@as(u16, 0xfffe)), 15);
    try expectEqual(countBitsSetParallel(@as(u16, 0xffff)), 16);
    try expectEqual(countBitsSetParallel(@as(u32, 0b0)), 0);
    try expectEqual(countBitsSetParallel(@as(u32, 0b11100011)), 5);
    try expectEqual(countBitsSetParallel(@as(u32, 0b11111111)), 8);
    try expectEqual(countBitsSetParallel(@as(u32, 0xfffffffe)), 31);
    try expectEqual(countBitsSetParallel(@as(u32, 0xffffffff)), 32);
    try expectEqual(countBitsSetParallel(@as(u64, 0x0)), 0);
    try expectEqual(countBitsSetParallel(@as(u64, 0x1)), 1);
    try expectEqual(countBitsSetParallel(@as(u64, 0xfffffffffffffffe)), 63);
    try expectEqual(countBitsSetParallel(@as(u64, 0xffffffffffffffff)), 64);
    try expectEqual(countBitsSetParallel(@as(u128, 0x0)), 0);
    try expectEqual(countBitsSetParallel(@as(u128, 0x1)), 1);
    try expectEqual(countBitsSetParallel(@as(u128, 0xfffffffffffffffffffffffffffffffe)), 127);
    try expectEqual(countBitsSetParallel(@as(u128, 0xffffffffffffffffffffffffffffffff)), 128);
}

/// Count bits set (rank) from the most-significant bit upto a given position
/// Returns rank of MSB bits in `val` downto LSB `pos`
/// https://github.com/cryptocode/bithacks#count-bits-set-rank-from-the-most-significant-bit-upto-a-given-position
pub fn countBitsRank(val: u64, pos: u64) u64 {
    const ones = ~@as(u64, 0);
    const bits = @as(u64, 64);

    // The following finds the the rank of a bit, meaning it returns the sum of bits that
    // are set to 1 from the most-signficant bit downto the bit at the given position.
    var r: u64 = val >> @as(u6, @intCast((bits -% pos)));
    r = r - ((r >> 1) & ones / 3);
    r = (r & ones / 5) + ((r >> 2) & ones / 5);
    r = (r +% (r >> 4)) & ones / 17;
    r = (r *% (ones / 255)) >> ((8 - 1) *% 8);
    return r;
}

test "Count bits set (rank) from the most-significant bit upto a given position" {
    try expectEqual((countBitsRank(0x0, 64)), 0);
    try expectEqual((countBitsRank(0x1, 64)), 1);
    try expectEqual((countBitsRank(0x1, 1)), 0);
    try expectEqual((countBitsRank(0xefffffffffffffff, 7)), 6);
    try expectEqual((countBitsRank(0xffffffffffffffff, 64)), 64);
}

/// Select the bit position (from the most-significant bit) with the given count `rank`
/// https://github.com/cryptocode/bithacks#SelectPosFromMSBRank
pub fn bitPosOfRank(val: u64, rank: u64) u64 {
    const ones = ~@as(u64, 0);

    // Do a normal parallel bit count for a 64-bit integer, but store all intermediate steps:
    const a: u64 = val - ((val >> 1) & ones / 3);
    const b: u64 = (a & ones / 5) + ((a >> 2) & ones / 5);
    const c: u64 = (b +% (b >> 4)) & ones / 0x11;
    const d: u64 = (c +% (c >> 8)) & ones / 0x101;
    var t: u64 = (d >> 32) + (d >> 48);
    var r = rank;

    // Now do branchless select:
    var s: u64 = 64;
    s -%= (t -% r) & 256 >> @as(u6, 3);
    r -%= (t & ((t -% r) >> 8));
    t = (d >> @as(u6, @intCast((s -% @as(u64, 16))))) & 0xff;
    s -%= ((t -% r) & 256) >> 4;
    r -%= (t & ((t -% r) >> 8));
    t = (c >> @as(u6, @intCast((s -% 8)))) & 0xf;
    s -%= ((t -% r) & 256) >> 5;
    r -%= (t & ((t -% r) >> 8));
    t = (b >> @as(u6, @intCast((s -% 4)))) & 0x7;
    s -%= ((t -% r) & 256) >> 6;
    r -%= (t & ((t -% r) >> 8));
    t = (a >> @as(u6, @intCast((s -% 2)))) & 0x3;
    s -%= ((t -% r) & 256) >> 7;
    r -%= (t & ((t -% r) >> 8));
    t = (val >> @as(u6, @intCast((s -% 1)))) & 0x1;
    s -%= ((t -% r) & 256) >> 8;
    s = 65 -% s;
    return s;
}

test "Select the bit position (from the most-significant bit) with the given count (rank)" {
    try expectEqual((bitPosOfRank(0xffffffffffffffff, 64)), 64);
    try expectEqual((bitPosOfRank(0x00ffffffffffffff, 1)), 9);
}

/// Computing parity the naive way
/// Returns true when an odd number of bits are set in `val`
/// https://github.com/cryptocode/bithacks#computing-parity-the-naive-way
pub fn parityNaive(val: anytype) bool {
    _ = requireInt(@TypeOf(val));
    var parity = false;

    var v = val;
    while (v != 0) {
        parity = !parity;
        v = v & (v - 1);
    }

    return parity;
}

test "Computing parity the naive way" {
    try expect(!parityNaive(@as(u8, 0x0)));
    try expect(!parityNaive(@as(u8, 0xf)));
    try expect(!parityNaive(@as(u8, 0xff)));
    try expect(parityNaive(@as(u8, 0x1)));
    try expect(parityNaive(@as(u8, 0x7)));
    try expect(parityNaive(@as(u32, 2)));
    try expect(parityNaive(@as(u32, 4)));
    try expect(parityNaive(@as(u32, 7)));
    try expect(!parityNaive(@as(u32, 0)));
    try expect(!parityNaive(@as(u32, 3)));
}

/// Compute parity by lookup table
/// Returns true when an odd number of bits are set in `val` which must be an 8-bit or 32-bit unsigned integer.
/// https://github.com/cryptocode/bithacks#compute-parity-by-lookup-table
pub fn parityByLookupTable(val: anytype) bool {
    const T = requireUnsignedInt(@TypeOf(val));
    comptime assert(@typeInfo(T).int.bits == 8 or @typeInfo(T).int.bits == 32);

    // Generate the lookup table at compile time which determines if the n'th number has an odd number of bits.
    // The table can be viewed as a 16 by 16 bit-matrix generated from a seed following these rules:
    // For each row n in [0..15], if the n'th bit in the seed is 0, use the seed as the row,
    // otherwise use the inverted seed as the row.
    const seed: u16 = 0b0110100110010110;
    const parityTable = comptime val: {
        var table: [16]u16 = undefined;
        var row: usize = 0;
        while (row < 16) : (row += 1) {
            table[row] = if (seed & (1 << (15 - row)) == 0) seed else ~seed;
        }

        break :val table;
    };

    var word = val / 16;
    var bit = val % 16;
    return 0 != switch (@typeInfo(T).int.bits) {
        8 => parityTable[word] & (@as(u16, 0x8000) >> @as(u4, @intCast(bit))),
        32 => res: {
            var v = val;
            v ^= v >> 16;
            v ^= v >> 8;

            const index = v & 0xff;
            word = index / 16;
            bit = index % 16;

            break :res parityTable[word] & (@as(u16, 0x8000) >> @as(u4, @intCast(bit)));
        },
        else => @panic("Invalid integer size"),
    };
}

test "Compute parity by lookup table" {
    try expect(parityByLookupTable(@as(u8, 0x1)));
    try expect(parityByLookupTable(@as(u8, 0x7)));
    try expect(!parityByLookupTable(@as(u8, 0x0)));
    try expect(!parityByLookupTable(@as(u8, 0xf)));
    try expect(!parityByLookupTable(@as(u8, 0xff)));
    try expect(parityByLookupTable(@as(u32, 2)));
    try expect(parityByLookupTable(@as(u32, 4)));
    try expect(parityByLookupTable(@as(u32, 7)));
    try expect(!parityByLookupTable(@as(u32, 0)));
    try expect(!parityByLookupTable(@as(u32, 3)));
}

/// Compute parity of a byte using 64-bit multiply and modulus division
/// https://github.com/cryptocode/bithacks#compute-parity-of-a-byte-using-64-bit-multiply-and-modulus-division
pub fn parityMulMod(val: u8) bool {
    return 0 != (((val * @as(u64, 0x0101010101010101)) & @as(u64, 0x8040201008040201)) % 0x1FF) & 1;
}

test "Compute parity of a byte using 64-bit multiply and modulus division" {
    try expect(!parityMulMod(0x0));
    try expect(!parityMulMod(0xf));
    try expect(!parityMulMod(0xff));
    try expect(parityMulMod(0x1));
    try expect(parityMulMod(0x7));
}

/// Compute parity of word with a multiply
/// The input `val` must be a 32 or 64 bit unsigned integer
/// https://github.com/cryptocode/bithacks#compute-parity-of-word-with-a-multiply
pub fn parityMul(val: anytype) bool {
    const T = requireUnsignedInt(@TypeOf(val));
    comptime assert(@typeInfo(T).int.bits == 32 or @typeInfo(T).int.bits == 64);

    return 0 != switch (@typeInfo(T).int.bits) {
        32 => res: {
            var v = val;
            v ^= v >> 1;
            v ^= v >> 2;
            v = (v & 0x11111111) *% 0x11111111;
            break :res (v >> 28) & 1;
        },
        64 => res: {
            var v = val;
            v ^= v >> 1;
            v ^= v >> 2;
            v = (v & 0x1111111111111111) *% 0x1111111111111111;
            break :res (v >> 60) & 1;
        },
        else => @panic("Invalid integer size"),
    };
}

test "Compute parity of word with a multiply" {
    try expect(parityMul(@as(u32, 2)));
    try expect(parityMul(@as(u32, 4)));
    try expect(parityMul(@as(u32, 7)));
    try expect(!parityMul(@as(u32, 0)));
    try expect(!parityMul(@as(u32, 3)));
    try expect(!parityMul(@as(u32, 0xffffffff)));
    try expect(parityMul(@as(u64, 2)));
    try expect(parityMul(@as(u64, 4)));
    try expect(parityMul(@as(u64, 7)));
    try expect(!parityMul(@as(u64, 0)));
    try expect(!parityMul(@as(u64, 3)));
    try expect(!parityMul(@as(u64, 0xffffffffffffffff)));
}

/// Compute parity in parallel
/// Works for 32-bit unsigned integers
pub fn parityParallel(val: u32) bool {
    var v = val;
    v ^= v >> 16;
    v ^= v >> 8;
    v ^= v >> 4;
    v &= 0xf;
    return 0 != ((@as(u16, 0x6996) >> @as(u4, @intCast(v))) & 1);
}

test "Compute parity in parallel" {
    try expect(parityParallel(2));
    try expect(parityParallel(4));
    try expect(parityParallel(7));
    try expect(!parityParallel(0));
    try expect(!parityParallel(3));
    try expect(!parityParallel(0xffffffff));
}

/// Swapping values with subtraction and addition
/// https://github.com/cryptocode/bithacks#swapping-values-with-subtraction-and-addition
pub fn swapSubAdd(a: anytype, b: anytype) void {
    if (a != b) {
        a.* -%= b.*;
        b.* +%= a.*;
        a.* = b.* -% a.*;
    }
}

test "Swapping values with subtraction and addition" {
    var a: u32 = 0x1dfa8ce1;
    var b: u32 = 0xffeeddcc;

    swapSubAdd(&a, &b);

    try expectEqual(a, 0xffeeddcc);
    try expectEqual(b, 0x1dfa8ce1);
}

/// Swapping values with XOR
/// https://github.com/cryptocode/bithacks#swapping-values-with-xor
pub fn swapXor(a: anytype, b: anytype) void {
    if (a != b) {
        a.* ^= b.*;
        b.* ^= a.*;
        a.* ^= b.*;
    }
}

test "Swapping values with XOR" {
    var a: u32 = 0x1dfa8ce1;
    var b: u32 = 0xffeeddcc;

    swapXor(&a, &b);

    try expectEqual(a, 0xffeeddcc);
    try expectEqual(b, 0x1dfa8ce1);
}

/// Swapping individual bits with XOR
/// https://github.com/cryptocode/bithacks#swapping-individual-bits-with-xor
pub fn swapBitsXor(pos1: usize, pos2: usize, consecutiveBits: usize, val: anytype) @TypeOf(val) {
    const T = requireInt(@TypeOf(val));
    const shiftType = std.math.Log2Int(T);

    const x: T = ((val >> @as(shiftType, @intCast(pos1))) ^ (val >> @as(shiftType, @intCast(pos2)))) & ((@as(T, 1) << @as(shiftType, @intCast(consecutiveBits))) - 1);
    return val ^ ((x << @as(shiftType, @intCast(pos1))) | (x << @as(shiftType, @intCast(pos2))));
}

test "Swapping individual bits with XOR" {
    try expectEqual(swapBitsXor(0, 4, 4, @as(u8, 0b11110000)), 0b00001111);
    try expectEqual(swapBitsXor(0, 16, 16, @as(u32, 0xffff0000)), 0x0000ffff);
}

/// Reverse bits the obvious way
/// https://github.com/cryptocode/bithacks#reverse-bits-the-obvious-way
pub fn reverseObvious(val: anytype) @TypeOf(val) {
    const T = requireInt(@TypeOf(val));
    const bits = @typeInfo(T).int.bits;
    const shiftType = std.math.Log2Int(T);

    var finalShiftsNeeded: shiftType = bits - 1;
    var v = val >> 1;
    var res = val;

    while (v != 0) {
        res <<= 1;
        res |= v & 1;
        finalShiftsNeeded -%= 1;
        v >>= 1;
    }

    return (res << finalShiftsNeeded);
}

test "Reverse bits the obvious way" {
    try expectEqual(reverseObvious(@as(u8, 0b11010010)), 0b01001011);
    try expectEqual(reverseObvious(@as(u8, 0b00000001)), 0b10000000);
    try expectEqual(reverseObvious(@as(u32, 0xfffffffe)), 0x7fffffff);
    try expectEqual(reverseObvious(@as(u32, 0xffffffff)), 0xffffffff);
    try expectEqual(reverseObvious(@as(u32, 0)), 0);
    try expectEqual(reverseObvious(@as(u32, 1)), 0x80000000);
    try expectEqual(reverseObvious(@as(u64, 0xfffffffffffffffe)), 0x7fffffffffffffff);
}

/// Reverse bits in word by lookup table
/// This is specific to 32-bit unsigned integers
/// https://github.com/cryptocode/bithacks#reverse-bits-in-word-by-lookup-table
pub fn reverseByLookup(val: u32) u32 {
    // Generate the lookup table at compile time. This corresponds to the macro-compacted C version.
    const reverseTable = comptime val: {
        var tblgen = struct {
            i: usize = 0,
            t: [256]u8 = undefined,

            pub fn R2(self: *@This(), n: u8) void {
                self.t[self.i + 0] = n;
                self.t[self.i + 1] = n + 2 * 64;
                self.t[self.i + 2] = n + 1 * 64;
                self.t[self.i + 3] = n + 3 * 64;
                self.i += 4;
            }
            pub fn R4(self: *@This(), n: u8) void {
                self.R2(n);
                self.R2(n + 2 * 16);
                self.R2(n + 1 * 16);
                self.R2(n + 3 * 16);
            }
            pub fn R6(self: *@This(), n: u8) void {
                self.R4(n);
                self.R4(n + 2 * 4);
                self.R4(n + 1 * 4);
                self.R4(n + 3 * 4);
            }
        }{};

        tblgen.R6(0);
        tblgen.R6(2);
        tblgen.R6(1);
        tblgen.R6(3);

        break :val tblgen.t;
    };

    return (@as(u32, @intCast(reverseTable[val & 0xff])) << 24) |
        (@as(u32, @intCast(reverseTable[(val >> 8) & 0xff])) << 16) |
        (@as(u32, @intCast(reverseTable[(val >> 16) & 0xff])) << 8) |
        (@as(u32, @intCast(reverseTable[(val >> 24) & 0xff])));
}

test "Reverse bits in word by lookup table" {
    try expectEqual(reverseByLookup(0xfffffffe), 0x7fffffff);
    try expectEqual(reverseByLookup(0xffffffff), 0xffffffff);
    try expectEqual(reverseByLookup(0), 0);
    try expectEqual(reverseByLookup(1), 0x80000000);
}

/// Reverse the bits in a byte with 3 operations (64-bit multiply and modulus division)
/// https://github.com/cryptocode/bithacks#reverse-the-bits-in-a-byte-with-3-operations-64-bit-multiply-and-modulus-division
pub fn reverseByteMulMod(val: u8) u8 {
    return @as(u8, @truncate((val * @as(u64, 0x0202020202) & @as(u64, 0x010884422010)) % 1023));
}

test "Reverse the bits in a byte with 3 operations (64-bit multiply and modulus division)" {
    try expectEqual(reverseByteMulMod(0b11010010), 0b01001011);
    try expectEqual(reverseByteMulMod(0b00000001), 0b10000000);
    try expectEqual(reverseByteMulMod(0), 0);
}

/// Reverse the bits in a byte with 4 operations (64-bit multiply, no division)
/// https://github.com/cryptocode/bithacks#reverse-the-bits-in-a-byte-with-4-operations-64-bit-multiply-no-division
pub fn reverseByteMulNoDiv(val: u8) u8 {
    return @as(u8, @truncate(((val * @as(u64, 0x80200802)) & @as(u64, 0x0884422110)) *% @as(u64, 0x0101010101) >> 32));
}

test "Reverse the bits in a byte with 4 operations (64-bit multiply, no division)" {
    try expectEqual(reverseByteMulNoDiv(0b11010010), 0b01001011);
    try expectEqual(reverseByteMulNoDiv(0b00000001), 0b10000000);
    try expectEqual(reverseByteMulNoDiv(0), 0);
}

/// Reverse the bits in a byte with 7 operations (no 64-bit)
/// https://github.com/cryptocode/bithacks#reverse-the-bits-in-a-byte-with-7-operations-no-64-bit
pub fn reverseByte7ops(val: u8) u8 {
    return @as(u8, @truncate(((val *% @as(u64, 0x0802) & @as(u64, 0x22110)) |
        (val *% @as(u64, 0x8020) & @as(u64, 0x88440))) *% @as(u64, 0x10101) >> 16));
}

test "Reverse the bits in a byte with 7 operations (no 64-bit)" {
    try expectEqual(reverseByte7ops(0b11010010), 0b01001011);
    try expectEqual(reverseByte7ops(0b00000001), 0b10000000);
    try expectEqual(reverseByte7ops(0), 0);
}

/// Reverse an N-bit quantity in parallel in 5 * lg(N) operations
/// https://github.com/cryptocode/bithacks#reverse-an-n-bit-quantity-in-parallel-in-5--lgn-operations
pub fn reverseInLog5steps(val: anytype) @TypeOf(val) {
    const T = requireInt(@TypeOf(val));
    const bits = @typeInfo(T).int.bits;
    comptime assert(std.math.isPowerOfTwo(bits));
    const shiftType = std.math.Log2Int(T);

    var v = val;
    var s: T = bits >> 1;
    var mask = ~@as(T, 0);

    while (s > 0) : (s >>= 1) {
        mask ^= (mask << @as(shiftType, @intCast(s)));
        v = ((v >> @as(shiftType, @intCast(s))) & mask) | ((v << @as(shiftType, @intCast(s))) & ~mask);
    }

    return v;
}

test "Reverse an N-bit quantity in parallel in 5 * lg(N) operations" {
    try expectEqual(reverseInLog5steps(@as(u32, 0xfffffffe)), 0x7fffffff);
    try expectEqual(reverseInLog5steps(@as(u32, 0xffffffff)), 0xffffffff);
    try expectEqual(reverseInLog5steps(@as(u32, 0)), 0);
    try expectEqual(reverseInLog5steps(@as(u32, 1)), 0x80000000);
    try expectEqual(reverseInLog5steps(@as(i32, 1)), -0x80000000);
    try expectEqual(reverseInLog5steps(@as(u64, 0xfffffffffffffffe)), 0x7fffffffffffffff);
}

/// Compute modulus division by 1 << s without a division operator
/// Returns `numerator` % (1 << `shiftAmount`), i.e. `numerator` % 2^n
/// https://github.com/cryptocode/bithacks#compute-modulus-division-by-1--s-without-a-division-operator
pub fn modPow2(numerator: anytype, shiftAmount: usize) @TypeOf(numerator) {
    const T = requireInt(@TypeOf(numerator));
    const shiftType = std.math.Log2Int(T);

    const d = @as(T, 1) << @as(shiftType, @intCast(shiftAmount));
    return numerator & (d - 1);
}

test "Compute modulus division by 1 << s without a division operator" {
    try expectEqual(modPow2(@as(u32, 19), 5), 19);
    try expectEqual(modPow2(@as(u32, 258), 8), 2);
    try expectEqual(modPow2(@as(i64, 19), 5), 19);
}

/// Compute modulus division by (1 << s) - 1 without a division operator
/// Returns `numerator` % ((1 << `shiftAmount`) - 1)
/// https://github.com/cryptocode/bithacks#compute-modulus-division-by-1--s---1-without-a-division-operator
pub fn modPow2Minus1(numerator: anytype, shiftAmount: usize) @TypeOf(numerator) {
    const T = requireInt(@TypeOf(numerator));
    const shiftType = std.math.Log2Int(T);

    const d = (@as(T, 1) << @as(shiftType, @intCast(shiftAmount))) - 1;
    var n = numerator;
    var m: T = numerator;
    while (n > d) : (n = m) {
        m = 0;
        while (n != 0) : (n >>= @as(shiftType, @intCast(shiftAmount))) {
            m +%= n & d;
        }
    }

    return if (m == d) 0 else m;
}

test "Compute modulus division by (1 << s) - 1 without a division operator" {
    try expectEqual(modPow2Minus1(@as(u8, 9), 3), 2);
    try expectEqual(modPow2Minus1(@as(u32, 9), 3), 2);
    try expectEqual(modPow2Minus1(@as(u32, 19), 3), 5);
    try expectEqual(modPow2Minus1(@as(u32, 21), 2), 0);
    try expectEqual(modPow2Minus1(@as(u64, 19), 3), 5);
}

/// Compute modulus division by (1 << s) - 1 in parallel without a division operator
/// https://github.com/cryptocode/bithacks#compute-modulus-division-by-1--s---1-in-parallel-without-a-division-operator
pub fn modPow2Minus1NoDiv(numerator: u32, shiftAmount: usize) u32 {
    // zig fmt: off
    const M: [32]u32 = .{
        0x00000000, 0x55555555, 0x33333333, 0xc71c71c7,  
        0x0f0f0f0f, 0xc1f07c1f, 0x3f03f03f, 0xf01fc07f, 
        0x00ff00ff, 0x07fc01ff, 0x3ff003ff, 0xffc007ff,
        0xff000fff, 0xfc001fff, 0xf0003fff, 0xc0007fff,
        0x0000ffff, 0x0001ffff, 0x0003ffff, 0x0007ffff, 
        0x000fffff, 0x001fffff, 0x003fffff, 0x007fffff,
        0x00ffffff, 0x01ffffff, 0x03ffffff, 0x07ffffff,
        0x0fffffff, 0x1fffffff, 0x3fffffff, 0x7fffffff
    };
    const Q: [32][6]u32 = .{
        .{ 0,  0,  0,  0,  0,  0}, .{16,  8,  4,  2,  1,  1}, .{16,  8,  4,  2,  2,  2},
        .{15,  6,  3,  3,  3,  3}, .{16,  8,  4,  4,  4,  4}, .{15,  5,  5,  5,  5,  5},
        .{12,  6,  6,  6 , 6,  6}, .{14,  7,  7,  7,  7,  7}, .{16,  8,  8,  8,  8,  8},
        .{ 9,  9,  9,  9,  9,  9}, .{10, 10, 10, 10, 10, 10}, .{11, 11, 11, 11, 11, 11},
        .{12, 12, 12, 12, 12, 12}, .{13, 13, 13, 13, 13, 13}, .{14, 14, 14, 14, 14, 14},
        .{15, 15, 15, 15, 15, 15}, .{16, 16, 16, 16, 16, 16}, .{17, 17, 17, 17, 17, 17},
        .{18, 18, 18, 18, 18, 18}, .{19, 19, 19, 19, 19, 19}, .{20, 20, 20, 20, 20, 20},
        .{21, 21, 21, 21, 21, 21}, .{22, 22, 22, 22, 22, 22}, .{23, 23, 23, 23, 23, 23},
        .{24, 24, 24, 24, 24, 24}, .{25, 25, 25, 25, 25, 25}, .{26, 26, 26, 26, 26, 26},
        .{27, 27, 27, 27, 27, 27}, .{28, 28, 28, 28, 28, 28}, .{29, 29, 29, 29, 29, 29},
        .{30, 30, 30, 30, 30, 30}, .{31, 31, 31, 31, 31, 31}            
    };
    const R: [32][6]u32 = .{
        .{0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
        .{0x0000ffff, 0x000000ff, 0x0000000f, 0x00000003, 0x00000001, 0x00000001},
        .{0x0000ffff, 0x000000ff, 0x0000000f, 0x00000003, 0x00000003, 0x00000003},
        .{0x00007fff, 0x0000003f, 0x00000007, 0x00000007, 0x00000007, 0x00000007},
        .{0x0000ffff, 0x000000ff, 0x0000000f, 0x0000000f, 0x0000000f, 0x0000000f},
        .{0x00007fff, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f, 0x0000001f},
        .{0x00000fff, 0x0000003f, 0x0000003f, 0x0000003f, 0x0000003f, 0x0000003f},
        .{0x00003fff, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f},
        .{0x0000ffff, 0x000000ff, 0x000000ff, 0x000000ff, 0x000000ff, 0x000000ff},
        .{0x000001ff, 0x000001ff, 0x000001ff, 0x000001ff, 0x000001ff, 0x000001ff}, 
        .{0x000003ff, 0x000003ff, 0x000003ff, 0x000003ff, 0x000003ff, 0x000003ff}, 
        .{0x000007ff, 0x000007ff, 0x000007ff, 0x000007ff, 0x000007ff, 0x000007ff}, 
        .{0x00000fff, 0x00000fff, 0x00000fff, 0x00000fff, 0x00000fff, 0x00000fff}, 
        .{0x00001fff, 0x00001fff, 0x00001fff, 0x00001fff, 0x00001fff, 0x00001fff}, 
        .{0x00003fff, 0x00003fff, 0x00003fff, 0x00003fff, 0x00003fff, 0x00003fff}, 
        .{0x00007fff, 0x00007fff, 0x00007fff, 0x00007fff, 0x00007fff, 0x00007fff}, 
        .{0x0000ffff, 0x0000ffff, 0x0000ffff, 0x0000ffff, 0x0000ffff, 0x0000ffff}, 
        .{0x0001ffff, 0x0001ffff, 0x0001ffff, 0x0001ffff, 0x0001ffff, 0x0001ffff}, 
        .{0x0003ffff, 0x0003ffff, 0x0003ffff, 0x0003ffff, 0x0003ffff, 0x0003ffff}, 
        .{0x0007ffff, 0x0007ffff, 0x0007ffff, 0x0007ffff, 0x0007ffff, 0x0007ffff},
        .{0x000fffff, 0x000fffff, 0x000fffff, 0x000fffff, 0x000fffff, 0x000fffff}, 
        .{0x001fffff, 0x001fffff, 0x001fffff, 0x001fffff, 0x001fffff, 0x001fffff}, 
        .{0x003fffff, 0x003fffff, 0x003fffff, 0x003fffff, 0x003fffff, 0x003fffff}, 
        .{0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff}, 
        .{0x00ffffff, 0x00ffffff, 0x00ffffff, 0x00ffffff, 0x00ffffff, 0x00ffffff},
        .{0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff, 0x01ffffff}, 
        .{0x03ffffff, 0x03ffffff, 0x03ffffff, 0x03ffffff, 0x03ffffff, 0x03ffffff}, 
        .{0x07ffffff, 0x07ffffff, 0x07ffffff, 0x07ffffff, 0x07ffffff, 0x07ffffff},
        .{0x0fffffff, 0x0fffffff, 0x0fffffff, 0x0fffffff, 0x0fffffff, 0x0fffffff},
        .{0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff, 0x1fffffff}, 
        .{0x3fffffff, 0x3fffffff, 0x3fffffff, 0x3fffffff, 0x3fffffff, 0x3fffffff}, 
        .{0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff}
    };
    // zig fmt: on

    const shiftType = std.math.Log2Int(u32);
    const s = shiftAmount;
    const d = (@as(u32, 1) << @as(shiftType, @intCast(shiftAmount))) - 1;
    const n = numerator;
    var m: u32 = (n & M[s]) +% ((n >> @as(shiftType, @intCast(s))) & M[s]);

    var q: usize = 0;
    var r: usize = 0;
    while (m > d) : ({
        q += 1;
        r += 1;
    }) {
        m = (m >> @as(shiftType, @intCast(Q[s][q]))) +% (m & R[s][r]);
    }
    return if (m == d) 0 else m;
}

test "Compute modulus division by (1 << s) - 1 in parallel without a division operator" {
    try expectEqual(modPow2Minus1NoDiv(9, 3), 2);
    try expectEqual(modPow2Minus1NoDiv(19, 3), 5);
    try expectEqual(modPow2Minus1NoDiv(21, 2), 0);
}

/// Find the log base 2 of an integer with the MSB N set in O(N) operations (the obvious way)
/// Returns ⌊log2(`val`)⌋, i.e. the position of the highest bit set.
/// https://github.com/cryptocode/bithacks#find-the-log-base-2-of-an-integer-with-the-msb-n-set-in-on-operations-the-obvious-way
pub fn log2floorObvious(val: anytype) @TypeOf(val) {
    const T = requireInt(@TypeOf(val));
    const shiftType = std.math.Log2Int(T);

    var v: T = val;
    var r: T = 0;
    while (true) {
        v >>= @as(shiftType, @intCast(1));
        if (v == 0) break;
        r +%= 1;
    }

    return r;
}

test "Find the log base 2 of an integer with the MSB N set in O(N) operations (the obvious way)" {
    try expectEqual(log2floorObvious(@as(u8, 127)), 6);
    try expectEqual(log2floorObvious(@as(u32, 0)), 0);
    try expectEqual(log2floorObvious(@as(u32, 1)), 0);
    try expectEqual(log2floorObvious(@as(u32, 2)), 1);
    try expectEqual(log2floorObvious(@as(u32, 127)), 6);
    try expectEqual(log2floorObvious(@as(u32, 128)), 7);
    try expectEqual(log2floorObvious(@as(u32, 0xffffffff)), 31);
    try expectEqual(log2floorObvious(@as(u64, 0xffffffffffffffff)), 63);
}

/// Find the integer log base 2 of an integer with an 64-bit IEEE float
/// Returns ⌊log2(`val`)⌋, i.e. the position of the highest bit set.
/// An improvement over the original is that 0 as input returns 0, and is thus consistent with `log2floorObvious`
/// https://github.com/cryptocode/bithacks#find-the-integer-log-base-2-of-an-integer-with-an-64-bit-ieee-float
pub fn log2usingFloat(val: u32) u32 {
    const endian = @import("builtin").target.cpu.arch.endian();
    const little_endian: bool = switch (endian) {
        .little => true,
        .big => false,
    };

    const U = extern union {
        u: [2]u32,
        d: f64,
    };

    if (val > 0) {
        var conv: U = undefined;
        conv.u[@intFromBool(little_endian)] = 0x43300000;
        conv.u[@intFromBool(!little_endian)] = val;
        conv.d -= 4503599627370496.0;
        return (conv.u[@intFromBool(little_endian)] >> 20) -% 0x3FF;
    } else {
        return 0;
    }
}

test "Find the integer log base 2 of an integer with an 64-bit IEEE float" {
    try expectEqual(log2usingFloat(0), 0);
    try expectEqual(log2usingFloat(1), 0);
    try expectEqual(log2usingFloat(2), 1);
    try expectEqual(log2usingFloat(127), 6);
    try expectEqual(log2usingFloat(128), 7);
    try expectEqual(log2usingFloat(0xffffffff), 31);
}

/// Find the log base 2 of an integer with a lookup table
/// Returns ⌊log2(`val`)⌋, i.e. the position of the highest bit set.
/// https://github.com/cryptocode/bithacks#find-the-log-base-2-of-an-integer-with-a-lookup-table
pub fn log2usingLookupTable(val: u32) u32 {

    // Build log table at compile time
    const logTable = comptime val: {
        var table: [256]u8 = undefined;
        table[0] = 0;
        table[1] = 0;
        var i: usize = 2;
        while (i < 256) : (i += 1) {
            table[i] = 1 + table[i / 2];
        }

        break :val table;
    };

    const tt: u32 = val >> 16;
    var t: u32 = undefined;

    if (tt != 0) {
        t = tt >> 8;
        return if (t != 0) 24 + logTable[t] else 16 + logTable[tt];
    } else {
        t = val >> 8;
        return if (t != 0) 8 + logTable[t] else logTable[val];
    }
}

test "Find the log base 2 of an integer with a lookup table" {
    try expectEqual(log2usingLookupTable(0), 0);
    try expectEqual(log2usingLookupTable(1), 0);
    try expectEqual(log2usingLookupTable(2), 1);
    try expectEqual(log2usingLookupTable(127), 6);
    try expectEqual(log2usingLookupTable(128), 7);
    try expectEqual(log2usingLookupTable(0xffffffff), 31);
}

/// Find the log base 2 of an N-bit integer in O(lg(N)) operations
/// https://github.com/cryptocode/bithacks#find-the-log-base-2-of-an-n-bit-integer-in-olgn-operations
pub fn log2inLogOps(val: u32) u32 {
    const shiftType = std.math.Log2Int(u32);

    const b: [5]u32 = .{ 0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000 };
    const S: [5]u32 = .{ 1, 2, 4, 8, 16 };
    var v = val;
    var i: i4 = 4;
    var res: u32 = 0;

    while (i >= 0) : (i -= 1) {
        const index = @as(usize, @intCast(i));
        if ((v & b[index]) != 0) {
            v >>= @as(shiftType, @intCast(S[index]));
            res |= S[index];
        }
    }

    return res;
}

test "Find the log base 2 of an N-bit integer in O(lg(N)) operations" {
    try expectEqual(log2inLogOps(0), 0);
    try expectEqual(log2inLogOps(1), 0);
    try expectEqual(log2inLogOps(2), 1);
    try expectEqual(log2inLogOps(127), 6);
    try expectEqual(log2inLogOps(128), 7);
    try expectEqual(log2inLogOps(0xffffffff), 31);
}

/// Find the log base 2 of an N-bit integer in O(lg(N)) operations with multiply and lookup
/// https://github.com/cryptocode/bithacks#find-the-log-base-2-of-an-n-bit-integer-in-olgn-operations-with-multiply-and-lookup
pub fn log2inLogOpsLookup(val: u32) u32 {
    // zig fmt: off
    const multiplyDeBruijnBitPosition: [32]u32 = .{ 
        0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 
        8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31 
    };
    // zig fmt: on

    var v = val;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;

    return multiplyDeBruijnBitPosition[@as(u32, (v *% @as(u32, 0x07C4ACDD))) >> 27];
}

test "Find the log base 2 of an N-bit integer in O(lg(N)) operations with multiply and lookup" {
    try expectEqual(log2inLogOpsLookup(0), 0);
    try expectEqual(log2inLogOpsLookup(1), 0);
    try expectEqual(log2inLogOpsLookup(2), 1);
    try expectEqual(log2inLogOpsLookup(127), 6);
    try expectEqual(log2inLogOpsLookup(128), 7);
    try expectEqual(log2inLogOpsLookup(0xffffffff), 31);
}

/// Find integer log base 10 of an integer
/// Returns 0 if `val` is 0, otherwise ⌊log10(`val`)⌋ is returned
/// https://github.com/cryptocode/bithacks#find-integer-log-base-10-of-an-integer
pub fn log10usingPowers(val: u32) u32 {
    if (val == 0) return 0;
    const powersOf10: [10]u32 =
        .{ 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000 };

    const t: u32 = (log2inLogOpsLookup(val) + 1) * 1233 >> 12; // (use a lg2 method from above)
    return t - @intFromBool(val < powersOf10[t]);
}

test "Find integer log base 10 of an integer" {
    try expectEqual(log10usingPowers(0), 0);
    try expectEqual(log10usingPowers(1), 0);
    try expectEqual(log10usingPowers(100), 2);
    try expectEqual(log10usingPowers(1000), 3);
    try expectEqual(log10usingPowers(1001), 3);
    try expectEqual(log10usingPowers(0xfffffff), 8);
    try expectEqual(log10usingPowers(0xffffffff), 9);
}

/// Find integer log base 10 of an integer the obvious way
/// https://github.com/cryptocode/bithacks#find-integer-log-base-10-of-an-integer-the-obvious-way
pub fn log10obvious(val: u32) u32 {
    // zig fmt: off
    return   if (val >= 1000000000) @as(u32, 9)
        else if (val >= 100000000) @as(u32, 8)
        else if (val >= 10000000) @as(u32, 7)
        else if (val >= 1000000) @as(u32, 6)
        else if (val >= 100000) @as(u32, 5)
        else if (val >= 10000) @as(u32, 4)
        else if (val >= 1000) @as(u32, 3)
        else if (val >= 100) @as(u32, 2)
        else if (val >= 10) @as(u32, 1)
        else 0;
    // zig fmt: on
}

test "Find integer log base 10 of an integer the obvious way" {
    try expectEqual(log10obvious(0), 0);
    try expectEqual(log10obvious(1), 0);
    try expectEqual(log10obvious(100), 2);
    try expectEqual(log10obvious(1000), 3);
    try expectEqual(log10obvious(1001), 3);
    try expectEqual(log10obvious(0xfffffff), 8);
    try expectEqual(log10obvious(0xffffffff), 9);
}

/// Find integer log base 2 of a 32-bit IEEE float
/// If `supportSubnormals` is true, a IEEE 754-compliant variant is used, otherwise a faster non-compliant variation is used
/// Returns ⌊log2(`val`)⌋
/// https://github.com/cryptocode/bithacks#find-integer-log-base-2-of-a-32-bit-ieee-float
pub fn log2float32(val: f32, comptime supportSubnormals: bool) u32 {
    if (val == 0) return 0;

    const U = extern union {
        f: f32,
        u: u32,
    };
    const conv: U = .{ .f = val };
    const x = conv.u;

    if (supportSubnormals) {
        // Build log table at compile time
        const logTable = comptime val: {
            var table: [256]u8 = undefined;
            table[0] = 0;
            table[1] = 0;
            var i: usize = 2;
            while (i < 256) : (i += 1) {
                table[i] = 1 + table[i / 2];
            }

            break :val table;
        };

        var c: u32 = x >> 23;

        if (c > 0) {
            c -%= 127;
        } else {
            // Subnormal, so recompute using mantissa: c = intlog2(x) - 149;
            var t: u32 = x >> 16;
            if (t > 0) {
                c = logTable[t] -% 133;
            } else {
                t = x >> 8;
                c = if (t > 0) logTable[t] -% 141 else logTable[x] -% 149;
            }
        }

        return c;
    } else {
        return (x >> 23) -% 127;
    }
}

test "Find integer log base 2 of a 32-bit IEEE float" {
    try expectEqual(log2float32(0, false), 0);
    try expectEqual(log2float32(1, false), 0);
    try expectEqual(log2float32(2, false), 1);
    try expectEqual(log2float32(127, false), 6);
    try expectEqual(log2float32(128, false), 7);

    try expectEqual(log2float32(0, true), 0);
    try expectEqual(log2float32(1, true), 0);
    try expectEqual(log2float32(2, true), 1);
    try expectEqual(log2float32(127, true), 6);
    try expectEqual(log2float32(128, true), 7);
}

/// Find integer log base 2 of the pow(2, r)-root of a 32-bit IEEE float (for unsigned integer r)
/// Input `val` must be have a normalized representation
/// https://github.com/cryptocode/bithacks#find-integer-log-base-2-of-the-pow2-r-root-of-a-32-bit-ieee-float-for-unsigned-integer-r
pub fn log2float32pow(val: f32, r: u32) u32 {
    assert(std.math.isNormal(val));
    const shiftType = std.math.Log2Int(u32);
    const U = extern union {
        f: f32,
        u: u32,
    };
    const conv: U = .{ .f = val };
    return ((((conv.u -% 0x3f800000) >> @as(shiftType, @intCast(r))) +% 0x3f800000) >> 23) -% 127;
}

test "Find integer log base 2 of the pow(2, r)-root of a 32-bit IEEE float (for unsigned integer r)" {
    try expectEqual(log2float32pow(16, 1), 2);
    try expectEqual(log2float32pow(1024, 3), 1);
}

/// Count the consecutive zero bits (trailing) on the right linearly
/// https://github.com/cryptocode/bithacks#count-the-consecutive-zero-bits-trailing-on-the-right-linearly
pub fn countConsecutiveZeroBitsLinearily(val: anytype) usize {
    const T = requireInt(@TypeOf(val));

    var v: T = val;
    var c: usize = undefined;
    if (v != 0) {
        v = (v ^ (v -% 1)) >> 1;
        c = 0;
        while (v != 0) : (c += 1) {
            v >>= 1;
        }
    } else {
        c = @typeInfo(T).int.bits;
    }

    return c;
}

test "Count the consecutive zero bits (trailing) on the right linearly" {
    try expectEqual(countConsecutiveZeroBitsLinearily(@as(u32, 104)), 3);
    try expectEqual(countConsecutiveZeroBitsLinearily(@as(u32, 0xffffffff)), 0);
    try expectEqual(countConsecutiveZeroBitsLinearily(@as(i32, 0x7fffffff)), 0);
    try expectEqual(countConsecutiveZeroBitsLinearily(@as(u8, 0)), 8);
    try expectEqual(countConsecutiveZeroBitsLinearily(@as(u32, 0)), 32);
    try expectEqual(countConsecutiveZeroBitsLinearily(@as(u64, 0)), 64);
    try expectEqual(countConsecutiveZeroBitsLinearily(@as(u128, 0)), 128);
}

/// Count the consecutive zero bits (trailing) on the right in parallel
/// https://github.com/cryptocode/bithacks#count-the-consecutive-zero-bits-trailing-on-the-right-in-parallel
pub fn countConsecutiveZeroBitsParallel(val: u32) usize {
    const v: u32 = val & -%val;
    var c: u32 = 32;

    if (v != 0) c -%= 1;
    if ((v & 0x0000FFFF) != 0) c -%= 16;
    if ((v & 0x00FF00FF) != 0) c -%= 8;
    if ((v & 0x0F0F0F0F) != 0) c -%= 4;
    if ((v & 0x33333333) != 0) c -%= 2;
    if ((v & 0x55555555) != 0) c -%= 1;
    return c;
}

test "Count the consecutive zero bits (trailing) on the right in parallel" {
    try expectEqual(countConsecutiveZeroBitsParallel(1), 0);
    try expectEqual(countConsecutiveZeroBitsParallel(104), 3);
    try expectEqual(countConsecutiveZeroBitsParallel(0xffffffff), 0);
    try expectEqual(countConsecutiveZeroBitsParallel(0), 32);
}

/// Count the consecutive zero bits (trailing) on the right by binary search
/// Input `val` must be non-zero
/// An improvement over the original is that a branch is eliminated if input is known to be
/// even through passing false to `canBeOdd`
/// https://github.com/cryptocode/bithacks#count-the-consecutive-zero-bits-trailing-on-the-right-by-binary-search
pub fn countConsecutiveZeroBitsBinarySearch(val: u32, comptime canBeOdd: bool) usize {
    var v: u32 = val;
    var c: u32 = 32;

    // If 0 == v, then c = 31.
    if (canBeOdd and v & 0x1 != 0) {
        // Special case for odd v (assumed to happen half of the time)
        c = 0;
    } else {
        c = 1;
        if ((v & 0xffff) == 0) {
            v >>= 16;
            c += 16;
        }
        if ((v & 0xff) == 0) {
            v >>= 8;
            c += 8;
        }
        if ((v & 0xf) == 0) {
            v >>= 4;
            c += 4;
        }
        if ((v & 0x3) == 0) {
            v >>= 2;
            c += 2;
        }
        c -= v & 0x1;
    }

    return c;
}

test "Count the consecutive zero bits (trailing) on the right by binary search" {
    try expectEqual(countConsecutiveZeroBitsBinarySearch(1, true), 0);
    try expectEqual(countConsecutiveZeroBitsBinarySearch(104, true), 3);
    try expectEqual(countConsecutiveZeroBitsBinarySearch(0xffffffff, true), 0);
}

/// Count the consecutive zero bits (trailing) on the right by casting to a float
/// https://github.com/cryptocode/bithacks#count-the-consecutive-zero-bits-trailing-on-the-right-by-casting-to-a-float
pub fn countConsecutiveZeroBitsUsingFloat(val: u32) usize {
    const U = extern union {
        f: f32,
        u: u32,
    };
    const conv: U = .{ .f = @as(f32, @floatFromInt(val & -%val)) };
    return (conv.u >> 23) - 0x7f;
}

test "Count the consecutive zero bits (trailing) on the right by casting to a float" {
    try expectEqual(countConsecutiveZeroBitsUsingFloat(1), 0);
    try expectEqual(countConsecutiveZeroBitsUsingFloat(104), 3);
    try expectEqual(countConsecutiveZeroBitsUsingFloat(0xffffffff), 0);
}

// Count the consecutive zero bits (trailing) on the right with modulus division and lookup
// https://github.com/cryptocode/bithacks#count-the-consecutive-zero-bits-trailing-on-the-right-with-modulus-division-and-lookup
pub fn countConsecutiveZeroBitsDivLookup(val: u32) usize {
    // zig fmt: off
    const mod37BitPosition: [37]u32 = .{ 
        32, 0, 1, 26, 2, 23, 27, 0, 3, 16, 24, 30, 28, 11, 0, 13, 4,
        7, 17, 0, 25, 22, 31, 15, 29, 10, 12, 6, 0, 21, 14, 9, 5,
        20, 8, 19, 18
    };
    // zig fmt: on
    return mod37BitPosition[(-%val & val) % 37];
}

test "Count the consecutive zero bits (trailing) on the right with modulus division and lookup" {
    try expectEqual(countConsecutiveZeroBitsDivLookup(1), 0);
    try expectEqual(countConsecutiveZeroBitsDivLookup(104), 3);
    try expectEqual(countConsecutiveZeroBitsDivLookup(0xffffffff), 0);
}

/// Count the consecutive zero bits (trailing) on the right with multiply and lookup
/// https://github.com/cryptocode/bithacks#count-the-consecutive-zero-bits-trailing-on-the-right-with-multiply-and-lookup
pub fn countConsecutiveZeroBitsMulLookup(val: u32) usize {
    // zig fmt: off
    const multiplyDeBruijnBitPosition: [32]u32 = .{ 
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9 };
    // zig fmt: on
    return multiplyDeBruijnBitPosition[((val & -%val) * @as(u32, 0x077CB531)) >> 27];
}

test "Count the consecutive zero bits (trailing) on the right with multiply and lookup" {
    try expectEqual(countConsecutiveZeroBitsMulLookup(1), 0);
    try expectEqual(countConsecutiveZeroBitsMulLookup(104), 3);
    try expectEqual(countConsecutiveZeroBitsMulLookup(0xffffffff), 0);
}

/// Round up to the next highest power of 2 by float casting
/// https://github.com/cryptocode/bithacks#round-up-to-the-next-highest-power-of-2-by-float-casting
pub fn roundToPow2ByFloat(val: u32) u32 {
    assert(val < (1 << 31));
    const shiftType = std.math.Log2Int(u32);
    if (val > 1) {
        const U = extern union {
            f: f32,
            u: u32,
        };
        const conv: U = .{ .f = @as(f32, @floatFromInt(val)) };
        const t = @as(u32, 1) << @as(shiftType, @intCast((conv.u >> 23) -% 0x7f));
        return t << @intFromBool(t < val);
    } else return 1;
}

test "Round up to the next highest power of 2 by float casting" {
    try expectEqual(roundToPow2ByFloat(0), 1);
    try expectEqual(roundToPow2ByFloat(3), 4);
    try expectEqual(roundToPow2ByFloat(7), 8);
    try expectEqual(roundToPow2ByFloat(8), 8);
    // Test highest supported input; higher inputs will assert
    try expectEqual(roundToPow2ByFloat((1 << 31) - 1), 0x80000000);
}

/// Round up to the next highest power of 2
/// https://github.com/cryptocode/bithacks#round-up-to-the-next-highest-power-of-2
pub fn roundToPow2By(val: u32) u32 {
    assert(val < (1 << 31));
    var v = val -% 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v +%= 1;
    // For consistency with `roundToPow2ByFlaot`, 0 => 1
    v +%= @intFromBool(v == 0);
    return v;
}

test "Round up to the next highest power of 2" {
    try expectEqual(roundToPow2By(0), 1);
    try expectEqual(roundToPow2By(3), 4);
    try expectEqual(roundToPow2By(7), 8);
    try expectEqual(roundToPow2By(8), 8);
    try expectEqual(roundToPow2ByFloat((1 << 31) - 1), 0x80000000);
}

fn DoubledIntSize(comptime T: type) type {
    return std.meta.Int(@typeInfo(T).int.signedness, @typeInfo(T).int.bits * 2);
}

/// Interleave bits the obvious way
/// Bits of `x` end up in even positions, bits of `y` in odd positions, and the interleaved result is returned
/// https://github.com/cryptocode/bithacks#interleave-bits-the-obvious-way
pub fn interleaveBitsObvious(first: anytype, second: @TypeOf(first)) DoubledIntSize(@TypeOf(first)) {
    const T = @TypeOf(first);
    const T2 = DoubledIntSize(T);
    const bits = @typeInfo(T).int.bits;
    const shiftType = std.math.Log2Int(T2);

    var res: T2 = 0;
    var i: isize = 0;
    while (i < bits) : (i += 1) {
        const i_shift = @as(shiftType, @intCast(i));
        res |= ((first & (@as(T2, 1) << i_shift)) << i_shift) | ((second & (@as(T2, 1) << i_shift)) << @as(shiftType, @intCast(i + 1)));
    }

    return res;
}

test "Interleave bits the obvious way" {
    try expectEqual(interleaveBitsObvious(@as(u16, 0), 0), 0);
    try expectEqual(interleaveBitsObvious(@as(u16, 1), 2), 9);
    try expectEqual(interleaveBitsObvious(@as(u16, 0xfefe), 0xfefe), 0xfffcfffc);
    try expectEqual(interleaveBitsObvious(@as(u16, std.math.maxInt(u16)), std.math.maxInt(u16)), std.math.maxInt(u32));
    try expectEqual(interleaveBitsObvious(@as(u32, std.math.maxInt(u32)), std.math.maxInt(u32)), std.math.maxInt(u64));
    try expectEqual(interleaveBitsObvious(@as(u64, std.math.maxInt(u64)), std.math.maxInt(u64)), std.math.maxInt(u128));
}

/// Interleave bits by table lookup
/// https://github.com/cryptocode/bithacks#interleave-bits-by-table-lookup
pub fn interleaveBitsLookup(x: u16, y: u16) u32 {
    // zig fmt: off
    const mortonTable256: [256]u32 = .{
        0x0000, 0x0001, 0x0004, 0x0005, 0x0010, 0x0011, 0x0014, 0x0015,
        0x0040, 0x0041, 0x0044, 0x0045, 0x0050, 0x0051, 0x0054, 0x0055,
        0x0140, 0x0141, 0x0144, 0x0145, 0x0150, 0x0151, 0x0154, 0x0155,
        0x0100, 0x0101, 0x0104, 0x0105, 0x0110, 0x0111, 0x0114, 0x0115,
        0x0400, 0x0401, 0x0404, 0x0405, 0x0410, 0x0411, 0x0414, 0x0415,
        0x0440, 0x0441, 0x0444, 0x0445, 0x0450, 0x0451, 0x0454, 0x0455,
        0x0500, 0x0501, 0x0504, 0x0505, 0x0510, 0x0511, 0x0514, 0x0515,
        0x0540, 0x0541, 0x0544, 0x0545, 0x0550, 0x0551, 0x0554, 0x0555,
        0x1000, 0x1001, 0x1004, 0x1005, 0x1010, 0x1011, 0x1014, 0x1015,
        0x1040, 0x1041, 0x1044, 0x1045, 0x1050, 0x1051, 0x1054, 0x1055,
        0x1100, 0x1101, 0x1104, 0x1105, 0x1110, 0x1111, 0x1114, 0x1115,
        0x1140, 0x1141, 0x1144, 0x1145, 0x1150, 0x1151, 0x1154, 0x1155,
        0x1400, 0x1401, 0x1404, 0x1405, 0x1410, 0x1411, 0x1414, 0x1415,
        0x1440, 0x1441, 0x1444, 0x1445, 0x1450, 0x1451, 0x1454, 0x1455,
        0x1500, 0x1501, 0x1504, 0x1505, 0x1510, 0x1511, 0x1514, 0x1515,
        0x1540, 0x1541, 0x1544, 0x1545, 0x1550, 0x1551, 0x1554, 0x1555,
        0x4000, 0x4001, 0x4004, 0x4005, 0x4010, 0x4011, 0x4014, 0x4015,
        0x4040, 0x4041, 0x4044, 0x4045, 0x4050, 0x4051, 0x4054, 0x4055,
        0x4100, 0x4101, 0x4104, 0x4105, 0x4110, 0x4111, 0x4114, 0x4115,
        0x4140, 0x4141, 0x4144, 0x4145, 0x4150, 0x4151, 0x4154, 0x4155,
        0x4400, 0x4401, 0x4404, 0x4405, 0x4410, 0x4411, 0x4414, 0x4415,
        0x4440, 0x4441, 0x4444, 0x4445, 0x4450, 0x4451, 0x4454, 0x4455,
        0x4500, 0x4501, 0x4504, 0x4505, 0x4510, 0x4511, 0x4514, 0x4515,
        0x4540, 0x4541, 0x4544, 0x4545, 0x4550, 0x4551, 0x4554, 0x4555,
        0x5000, 0x5001, 0x5004, 0x5005, 0x5010, 0x5011, 0x5014, 0x5015,
        0x5040, 0x5041, 0x5044, 0x5045, 0x5050, 0x5051, 0x5054, 0x5055,
        0x5100, 0x5101, 0x5104, 0x5105, 0x5110, 0x5111, 0x5114, 0x5115,
        0x5140, 0x5141, 0x5144, 0x5145, 0x5150, 0x5151, 0x5154, 0x5155,
        0x5400, 0x5401, 0x5404, 0x5405, 0x5410, 0x5411, 0x5414, 0x5415,
        0x5440, 0x5441, 0x5444, 0x5445, 0x5450, 0x5451, 0x5454, 0x5455,
        0x5500, 0x5501, 0x5504, 0x5505, 0x5510, 0x5511, 0x5514, 0x5515,
        0x5540, 0x5541, 0x5544, 0x5545, 0x5550, 0x5551, 0x5554, 0x5555
    };
    // zig fmt: on

    return mortonTable256[y >> 8] << 17 |
        mortonTable256[x >> 8] << 16 |
        mortonTable256[y & 0xFF] << 1 |
        mortonTable256[x & 0xFF];
}

test "Interleave bits by table lookup" {
    try expectEqual(interleaveBitsLookup(0, 0), 0);
    try expectEqual(interleaveBitsLookup(1, 2), 9);
    try expectEqual(interleaveBitsLookup(0xfefe, 0xfefe), 0xfffcfffc);
    try expectEqual(interleaveBitsLookup(std.math.maxInt(u16), std.math.maxInt(u16)), std.math.maxInt(u32));
}

/// Interleave bits with 64-bit multiply
/// https://github.com/cryptocode/bithacks#interleave-bits-with-64-bit-multiply
pub fn interleaveBitsMul(first: u8, second: u8) u16 {
    const x: u16 = first;
    const y: u16 = second;

    return @as(u16, @truncate(((((x *%
        @as(u64, 0x0101010101010101)) & 0x8040201008040201) *%
        @as(u64, 0x0102040810204081)) >> 49) & 0x5555 | (((((y *%
        @as(u64, 0x0101010101010101)) & 0x8040201008040201) *%
        @as(u64, 0x0102040810204081)) >> 48) & 0xaaaa)));
}

test "Interleave bits with 64-bit multiply" {
    try expectEqual(interleaveBitsMul(0, 0), 0);
    try expectEqual(interleaveBitsMul(1, 2), 9);
    try expectEqual(interleaveBitsMul(0xfe, 0xfe), 0xfffc);
    try expectEqual(interleaveBitsMul(std.math.maxInt(u8), std.math.maxInt(u8)), std.math.maxInt(u16));
}

/// Interleave bits by Binary Magic Numbers
/// https://github.com/cryptocode/bithacks#interleave-bits-by-binary-magic-numbers
pub fn interleaveBitsMagic(first: u16, second: u16) u32 {
    const B: [4]u32 = .{ 0x55555555, 0x33333333, 0x0f0f0f0f, 0x00ff00ff };
    const S: [4]u32 = .{ 1, 2, 4, 8 };

    var x: u32 = first;
    var y: u32 = second;

    x = (x | (x << S[3])) & B[3];
    x = (x | (x << S[2])) & B[2];
    x = (x | (x << S[1])) & B[1];
    x = (x | (x << S[0])) & B[0];

    y = (y | (y << S[3])) & B[3];
    y = (y | (y << S[2])) & B[2];
    y = (y | (y << S[1])) & B[1];
    y = (y | (y << S[0])) & B[0];

    return x | (y << 1);
}

test "Interleave bits by Binary Magic Numbers" {
    try expectEqual(interleaveBitsMagic(0, 0), 0);
    try expectEqual(interleaveBitsMagic(1, 2), 9);
    try expectEqual(interleaveBitsMagic(0xfefe, 0xfefe), 0xfffcfffc);
    try expectEqual(interleaveBitsMagic(std.math.maxInt(u16), std.math.maxInt(u16)), std.math.maxInt(u32));
}

/// Determine if a word has a zero byte
/// https://github.com/cryptocode/bithacks#determine-if-a-word-has-a-zero-byte
pub fn wordContainsZeroByte(val: u32) bool {
    return (((val -% 0x01010101) & ~val) & 0x80808080) != 0;
}

test "Determine if a word has a zero byte" {
    try expect(wordContainsZeroByte(0xff00ffff));
    try expect(wordContainsZeroByte(0));
    try expect(!wordContainsZeroByte(0xffffffff));
}

/// Determine if a word has a byte equal to `needle`
/// https://github.com/cryptocode/bithacks#determine-if-a-word-has-a-byte-equal-to-n
pub fn wordContainsByte(val: u32, needle: u8) bool {
    return wordContainsZeroByte(val ^ (~@as(u32, 0) / 255 * needle));
}

test "Determine if a word has a byte equal to n" {
    try expect(wordContainsByte(0xff000000, 0xff));
    try expect(wordContainsByte(0x00ff0000, 0xff));
    try expect(wordContainsByte(0x0000ff00, 0xff));
    try expect(wordContainsByte(0x000000ff, 0xff));
    try expect(wordContainsByte(0xff001c00, 0x1c));
    try expect(!wordContainsByte(0xff001c00, 0xec));
}

/// Determine if a word has a byte less than `n` (which must be <= 128)
/// https://github.com/cryptocode/bithacks#determine-if-a-word-has-a-byte-less-than-n
pub fn wordContainsByteLessThan(val: anytype, n: u8) bool {
    assert(n <= @as(u8, 128));
    const T = requireUnsignedInt(@TypeOf(val));
    return (((val -% ((~@as(T, 0) / 255) *% n)) & ~val) & ((~@as(T, 0) / 255) *% 128)) != 0;
}

/// Counts the number of bytes in x that are less than n
/// https://github.com/cryptocode/bithacks#determine-if-a-word-has-a-byte-less-than-n
pub fn countBytesLessThan(val: anytype, n: u8) usize {
    assert(n <= @as(u8, 128));
    const T = requireUnsignedInt(@TypeOf(val));

    const maxBy255 = ~@as(T, 0) / 255;
    const res = (((((maxBy255 *% @as(T, 127 +% n)) -%
        (val & (maxBy255 *% 127))) & ~val) &
        (maxBy255 *% 128)) / 128) % 255;
    return @as(usize, @intCast(res));
}

test "Determine if a word has a byte less than n" {
    // Containment tests
    try expect(wordContainsByteLessThan(@as(u32, 0), 1));
    try expect(wordContainsByteLessThan(@as(u32, 0xff79ffff), 0x80));
    try expect(!wordContainsByteLessThan(@as(u32, 0xffffffff), 0x80));
    try expect(wordContainsByteLessThan(@as(u64, 0xff79ffffffffffff), 0x80));

    // Counting tests
    try expectEqual(countBytesLessThan(@as(u32, 0), 1), 4);
    try expectEqual(countBytesLessThan(@as(u64, 0), 1), 8);
    try expectEqual(countBytesLessThan(@as(u128, 0), 1), 16);
    try expectEqual(countBytesLessThan(@as(u32, 0xff79ffff), 0x80), 1);
    try expectEqual(countBytesLessThan(@as(u32, 0xffffffff), 0x80), 0);
}

/// Determine if a word has a byte greater than n
/// https://github.com/cryptocode/bithacks#determine-if-a-word-has-a-byte-greater-than-n
pub fn wordContainsByteGreaterThan(val: anytype, n: u8) bool {
    assert(n <= @as(u8, 127));
    const T = requireUnsignedInt(@TypeOf(val));
    return (((val +% ((~@as(T, 0) / 255) *% (127 -% n))) | val) & ((~@as(T, 0) / 255) *% 128)) != 0;
}

/// Counts the number of bytes in x that are less than n where n <= 127
pub fn countBytesGreaterThan(val: anytype, n: u8) usize {
    assert(n <= @as(u8, 127));
    const T = requireUnsignedInt(@TypeOf(val));

    const maxBy255 = ~@as(T, 0) / 255;
    const res = (((((val & (maxBy255 *% 127)) +%
        (maxBy255 *% (127 -% n))) | val) &
        (maxBy255 *% 128)) / 128) % 255;
    return @as(usize, @intCast(res));
}

test "Determine if a word has a byte greater than n" {
    // Containment tests
    try expect(!wordContainsByteGreaterThan(@as(u32, 0), 1));
    try expect(wordContainsByteGreaterThan(@as(u32, 0x00810000), 0x7F));
    try expect(wordContainsByteGreaterThan(@as(u64, 0x0081000000000000), 0x7F));

    // Counting tests
    try expectEqual(countBytesGreaterThan(@as(u32, std.math.maxInt(u32)), 1), 4);
    try expectEqual(countBytesGreaterThan(@as(u64, std.math.maxInt(u64)), 1), 8);
    try expectEqual(countBytesGreaterThan(@as(u128, std.math.maxInt(u128)), 1), 16);
    try expectEqual(countBytesGreaterThan(@as(u32, 0x00800000), 0x7F), 1);
    try expectEqual(countBytesGreaterThan(@as(u32, 0x0), 0x7F), 0);
}

/// Helper to implement both predicate and counting range test
fn wordHasByteBetweenNumericResult(val: anytype, m: u8, n: u8) @TypeOf(val) {
    assert(m <= @as(u8, 127));
    assert(n <= @as(u8, 128));
    const T = requireUnsignedInt(@TypeOf(val));

    const maxBy255 = ~@as(T, 0) / 255;
    return (((((maxBy255 *% (127 +% n)) -%
        (val & (maxBy255 *% 127))) & ~val) &
        ((val & (maxBy255 *% 127)) +% (maxBy255 *% (127 -% m)))) &
        (maxBy255 *% 128));
}

/// Determine if a word has a byte between m and n, where `m` <= 127 and `n` <= 128
/// https://github.com/cryptocode/bithacks#determine-if-a-word-has-a-byte-between-m-and-n
pub fn wordHasByteBetween(val: anytype, m: u8, n: u8) bool {
    return wordHasByteBetweenNumericResult(val, m, n) != 0;
}

/// Count the number of bytes in `val` that are between m and n (exclusive),
/// where `m` <= 127 and `n` <= 128
/// https://github.com/cryptocode/bithacks#determine-if-a-word-has-a-byte-between-m-and-n
pub fn countBytesBetween(val: anytype, m: u8, n: u8) @TypeOf(val) {
    return wordHasByteBetweenNumericResult(val, m, n) / 128 % 255;
}

test "Determine if a word has a byte between m and n" {
    try expect(!wordHasByteBetween(@as(u32, 0), 1, 128));
    try expect(!wordHasByteBetween(@as(u32, 0x00070000), 0x01, 0x06));
    try expect(wordHasByteBetween(@as(u32, 0x00050000), 0x01, 0x06));
    try expect(wordHasByteBetween(@as(u64, 0x0005000000000000), 0x01, 0x06));
    // Make sure upper bound is exclusive
    try expectEqual(countBytesBetween(@as(u64, 0x001a00001b001c1d), 0x01, 0x1d), 3);
    try expectEqual(countBytesBetween(@as(u64, 0x00), 0, 128), 0);
    try expectEqual(countBytesBetween(@as(u64, 0x01), 0, 128), 1);
    try expectEqual(countBytesBetween(@as(u8, 0x01), 0, 128), 1);
    try expectEqual(countBytesBetween(@as(u16, 0x0101), 0, 128), 2);
}

/// Compute the lexicographically next bit permutation
/// Given an initial `val` with N bits set, this function returns the next number
/// that also has N bits set.
/// The result is undefined if `val` is already the largest possible permutation.
/// https://github.com/cryptocode/bithacks#compute-the-lexicographically-next-bit-permutation
pub fn nextLexicographicPermutation(val: u32) u32 {
    // Input's least significant 0 bits set to 1
    const t = val | (val - 1);

    // Set to 1 the most significant bit to change, set to 0 the least significant ones, and add the necessary 1 bits.
    return (t + 1) | (((~t & -%~t) - 1) >> @as(u5, @intCast(@ctz(val) + 1)));
}

test "Compute the lexicographically next bit permutation" {
    try expectEqual(nextLexicographicPermutation(@as(u32, 0b00000001)), 0b00000010);
    try expectEqual(nextLexicographicPermutation(@as(u32, 0b00010011)), 0b00010101);
    try expectEqual(nextLexicographicPermutation(@as(u32, 0b00010101)), 0b00010110);
    try expectEqual(nextLexicographicPermutation(@as(u32, 0b00010110)), 0b00011001);
    try expectEqual(nextLexicographicPermutation(@as(u32, 0b00011001)), 0b00011010);
    try expectEqual(nextLexicographicPermutation(@as(u32, 0b00011010)), 0b00011100);
    try expectEqual(nextLexicographicPermutation(@as(u32, 0b00011100)), 0b00100011);
}

/// Clear the lowest set bit. The optimizer seems to correctly lower this to blsr and equivalent instructions.
/// (This is an addition to the original bithacks document)
pub fn clearLowestSetBit(val: anytype) @TypeOf(val) {
    return val & (val - 1);
}

test "Clear least significant set bit " {
    try expectEqual(clearLowestSetBit(@as(u32, 0b00000001)), 0b00000000);
    try expectEqual(clearLowestSetBit(@as(u32, 0b00011010)), 0b00011000);
    try expectEqual(clearLowestSetBit(@as(u64, 0b00000001)), 0b00000000);
    try expectEqual(clearLowestSetBit(@as(u64, 0b000110110001101100011011000110110001101100011000)), 0b000110110001101100011011000110110001101100010000);
}
