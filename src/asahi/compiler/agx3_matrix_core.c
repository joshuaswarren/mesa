#include "agx3_matrix_core.h"
#include "agx3_binder.h"

bool
agx3_matrix_encode_mma8(enum agx_isa_encoding encoding,
                        const struct agx3_matrix_mma8_desc *d,
                        uint8_t out[14], uint8_t *length)
{
   if (!d)
      return false;
   const bool agx3b = encoding == AGX_ISA_ENCODING_AGX3B;
   unsigned type = d->input_type ? d->input_type : AGX3_MATRIX_INPUT_F16;
   unsigned align_mask = type == AGX3_MATRIX_INPUT_F16 ? 3 : 1;
   unsigned dest_max = agx3b ? 188 : 252;
   unsigned source_max =
      type == AGX3_MATRIX_INPUT_F32 ? dest_max : (agx3b ? 190 : 254);

   /*
    * The 8x8 matrix multiply-add (f16/bf16/f32 inputs, f32 accumulator)
    * exists both on the M5 AGX3C encoding (G17G/G17S/G17C) and on the AGX3B
    * encoding that covers G15/G16 and G17P (M4 Pro, M4 Max, A18 Pro). The
    * shared AGX3_MN_f32matmad8x8 form carries every input type in the type field, so
    * one builder serves both; only the general register file ceiling (192
    * halves on AGX3B vs 256 on AGX3C), the witnessed selector constants and
    * the scoreboard-wait field range differ between the two encodings. The
    * 16x16 GEMM form (agx3_matrix_encode_gemm16) is M5 only, so this is the
    * sole matrix instruction AGX3B offers.
    */
   if ((encoding != AGX_ISA_ENCODING_AGX3C && !agx3b) || !out || !length ||
       (type != AGX3_MATRIX_INPUT_F16 && type != AGX3_MATRIX_INPUT_F32 &&
        type != AGX3_MATRIX_INPUT_BF16) ||
       (d->dest & align_mask) || d->dest > dest_max ||
       (d->a & 1) || d->a > source_max || (d->b & 1) || d->b > source_max ||
       (!d->c_zero && ((d->c & align_mask) || d->c > dest_max)) ||
       (d->c_zero && (d->c || d->cache[2] || d->discard[2])))
      return false;

   /*
    * AGX3B carries a six-bit scoreboard wait in fields 69..74 (slots 0..5):
    * field 74 is wait slot 5, not a c-is-zero marker. The c-is-zero vs
    * multiply-accumulate choice is field 59 alone (1 = no C operand,
    * 2 = multiply-accumulate), with the C register zero when absent. Fields
    * 77..79 are not part of the wait mask and stay clear (no Apple witness),
    * so refuse a wait above slot 5 rather than guess. AGX3C keeps its full
    * eight-bit wait in 69..76. Witnessed by Apple G16G compiler output; the
    * earlier "field 74 == c_zero, wait bits 0..4" reading byte-matched the
    * compose corpus only because those captures all set wait slot 5.
    */
   if (agx3b && (d->wait_mask & ~0x3fu))
      return false;

   for (unsigned i = 0; i < 3; ++i) {
      if (d->cache[i] && d->discard[i])
         return false;
   }

   struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 1}, {42, 2}, {43, 1},
      {60, 2}, {67, 3}, {68, 61},
   };
   /* f67/f68 are AGX3C-only selector bits; AGX3B does not carry them. */
   unsigned nbase = agx3b ? 7 : (sizeof(base) / sizeof(base[0]));
   struct agx3_field_kv fields[24];
   unsigned count = 0;
   fields[count++] = (struct agx3_field_kv){26, type};
   fields[count++] = (struct agx3_field_kv){43, type};
   fields[count++] = (struct agx3_field_kv){2, d->dest};
   fields[count++] = (struct agx3_field_kv){11, d->dest_cache};
   fields[count++] = (struct agx3_field_kv){18, d->a};
   fields[count++] = (struct agx3_field_kv){35, d->b};
   fields[count++] = (struct agx3_field_kv){52, d->c};
   fields[count++] = (struct agx3_field_kv){59, d->c_zero ? 1 : 2};
   fields[count++] = (struct agx3_field_kv){27, d->cache[0]};
   fields[count++] = (struct agx3_field_kv){29, d->discard[0]};
   fields[count++] = (struct agx3_field_kv){44, d->cache[1]};
   fields[count++] = (struct agx3_field_kv){46, d->discard[1]};
   fields[count++] = (struct agx3_field_kv){61, d->cache[2]};
   fields[count++] = (struct agx3_field_kv){63, d->discard[2]};
   for (unsigned i = 0; i < (agx3b ? 6u : 8u); ++i)
      fields[count++] = (struct agx3_field_kv){69 + i,
                                             (d->wait_mask >> i) & 1};

   return agx3_bind_compose(encoding, AGX3_MN_f32matmad8x8, 0, base,
                            nbase, fields, count,
                            out, length);
}

bool
agx3_matrix_encode_gemm16(enum agx_isa_encoding encoding,
                         const struct agx3_matrix_gemm16_desc *d,
                         uint8_t out[10], uint8_t *length)
{
   if (!d)
      return false;
   enum agx3_matrix_input_type b_type = d->b_input_type ? d->b_input_type : d->input_type;
   if (encoding != AGX_ISA_ENCODING_AGX3C || !out || !length ||
       (b_type != AGX3_MATRIX_INPUT_F16 && b_type != AGX3_MATRIX_INPUT_BF16) ||
       (d->dest & 15) || d->dest > 240 ||
       (d->a & 7) || d->a > 248 || (d->b & 7) || d->b > 248 ||
       (d->input_type != AGX3_MATRIX_INPUT_F16 &&
        d->input_type != AGX3_MATRIX_INPUT_BF16))
      return false;

   const struct agx3_field_kv base[] = {
      {1, 2}, {2, 2}, {9, 2}, {14, 1}, {20, 2}, {25, 1},
      {30, 1}, {31, 3}, {32, 70},
   };
   struct agx3_field_kv fields[29];
   unsigned count = 0;
   fields[count++] = (struct agx3_field_kv){0, d->dest};
   fields[count++] = (struct agx3_field_kv){3, d->dest_hint};
   fields[count++] = (struct agx3_field_kv){8, d->a};
   fields[count++] = (struct agx3_field_kv){10, d->input_type};
   fields[count++] = (struct agx3_field_kv){19, d->b};
   fields[count++] = (struct agx3_field_kv){21, b_type};
   fields[count++] = (struct agx3_field_kv){7, d->zero_accumulator};
   fields[count++] = (struct agx3_field_kv){15, d->transpose_a};
   fields[count++] = (struct agx3_field_kv){14, !d->preserve_a};
   fields[count++] = (struct agx3_field_kv){25, !d->preserve_b};
   fields[count++] = (struct agx3_field_kv){26, d->transpose_b};
   fields[count++] = (struct agx3_field_kv){33, d->field33};
   for (unsigned i = 0; i < 8; ++i)
      fields[count++] = (struct agx3_field_kv){40 + i,
                                             (d->field40_mask >> i) & 1};
   for (unsigned i = 0; i < 8; ++i)
      if (d->wait_mask & (1u << i))
         fields[count++] = (struct agx3_field_kv){33 + i, 1};
   return agx3_bind_compose(encoding, AGX3_MN_fGeMM16x16x16, 0, base,
                            sizeof(base) / sizeof(base[0]), fields, count,
                            out, length);
}
