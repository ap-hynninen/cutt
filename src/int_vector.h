#ifndef INT_VECTOR_H
#define INT_VECTOR_H

// Intel: Minimum SSE2 required. SSE can't be used because it does not support integer operations

#if defined(__SSE2__)
// Intel x86
#include <x86intrin.h>

#if defined(__AVX__)
const int INT_VECTOR_LEN = 8;

#if defined(__AVX2__)
const char INT_VECTOR_TYPE[] = "AVX2";
#else
const char INT_VECTOR_TYPE[] = "AVX";
#endif

#else
const int INT_VECTOR_LEN = 4;
const char INT_VECTOR_TYPE[] = "SSE2";
#endif

#elif defined(__ALTIVEC__)
// IBM altivec
#include <altivec.h>
const int INT_VECTOR_LEN = 4;
const char INT_VECTOR_TYPE[] = "ALTIVEC";

#else
// Nothing
const int INT_VECTOR_LEN = 1;
const char INT_VECTOR_TYPE[] = "SCALAR";
#endif

#ifdef DEBUG_INT_VECTOR
class int_vector;
void print_int32(const int_vector a);
#endif

//
// Integer vector class for Intel and IBM CPU platforms
//
class int_vector {
private:

#if defined(__AVX__)
  __m256i x;
#elif defined(__SSE2__)
  __m128i x;
#elif defined(__ALTIVEC__)
  vector signed int;
#else
  int x;
#endif

public:

  inline int_vector(const int a) {
#if defined(__AVX__)
    x = _mm256_set1_epi32(a);
#elif defined(__SSE2__)
    x = _mm_set1_epi32(a);
#elif defined(__ALTIVEC__)
    x = a;
#else
    x = a;
#endif    
  }

  inline int_vector(const int a[]) {
#if defined(__AVX__)
    x = _mm256_set_epi32(a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0]);
#elif defined(__SSE2__)
    x = _mm_set_epi32(a[3], a[2], a[1], a[0]);
#elif defined(__ALTIVEC__)
#else
    x = a[0];
#endif    
  }

#if defined(__AVX__)
  inline int_vector(const __m256i ax) {
    x = ax;
  }
#elif defined(__SSE2__)
  inline int_vector(const __m128i ax) {
    x = ax;
  }
#endif

  // 
  // Member functions
  //
  inline int_vector operator*=(const int_vector a) {
#if defined(__AVX__)
    x = _mm256_mullo_epi32(x, a.x);
#elif defined(__SSE2__)
    x = _mm_mullo_epi32(x, a.x);
#elif defined(__ALTIVEC__)
#else
    x = x*a.x;
#endif
    return *this;
  }

  inline int_vector operator+=(const int_vector a) {
#if defined(__AVX__)
    x = _mm256_add_epi32(x, a.x);
#elif defined(__SSE2__)
    x = _mm_add_epi32(x, a.x);
#elif defined(__ALTIVEC__)
#else
    x = x + a.x;
#endif
    return *this;
  }

  inline int_vector operator-=(const int_vector a) {
#if defined(__AVX__)
    x = _mm256_sub_epi32(x, a.x);
#elif defined(__SSE2__)
    x = _mm_sub_epi32(x, a.x);
#elif defined(__ALTIVEC__)
#else
    x = x - a.x;
#endif
    return *this;
  }

  inline int_vector operator&=(const int_vector a) {
#if defined(__AVX__)
    x = _mm256_and_si256(x, a.x);
#elif defined(__SSE2__)
    x = _mm_and_si128(x, a.x);
#elif defined(__ALTIVEC__)
#else
    x = x & a.x;
#endif
    return *this;
  }

  inline int_vector operator~() {
#if defined(__AVX__)
    int_vector fullmask = int_vector(-1);
    x = _mm256_andnot_si256(x, fullmask.x);
#elif defined(__SSE2__)
    int_vector fullmask = int_vector(-1);
    x = _mm_andnot_si128(x, fullmask.x);
#elif defined(__ALTIVEC__)
#else
    x = ~x;
#endif
    return *this;
  }

  // Sign extended shift by a constant
  inline int_vector operator>>=(const int n) {
#if defined(__AVX__)
    x = _mm256_srai_epi32(x, n);
#elif defined(__SSE2__)
    x = _mm_srai_epi32(x, n);
#elif defined(__ALTIVEC__)
#else
    x >>= n;
#endif
    return *this;
  }

  // Sign extended shift by different amounts
  inline int_vector operator>>=(const int_vector n) {
#if defined(__AVX2__)
    x = _mm256_srav_epi32(x, n.x);
#elif defined(__AVX__)
#error "AVX not implemented: int_vector operator>>=(const int_vector n)"
#elif defined(__SSE2__)
#error "SSE2 not implemented: int_vector operator>>=(const int_vector n)"
#elif defined(__ALTIVEC__)
#else
    x >>= n;
#endif
    return *this;
  }

  inline int operator[](const int i) const {
#if defined(__AVX__)
    int res[INT_VECTOR_LEN];
    _mm256_storeu_si256((__m256i *)&res, x);
    return res[i % INT_VECTOR_LEN];
#elif defined(__SSE2__)
    int res[INT_VECTOR_LEN];
    _mm_storeu_si128((__m128i *)&res, x);
    return res[i % INT_VECTOR_LEN];
#elif defined(__ALTIVEC__)
    return 1;
#else
    return x;
#endif
  }

  inline int operator[](const int i) {
    return ((const int_vector)x)[i];
  }

  // Copy contest to int array
  void copy(int* a) const {
#if defined(__AVX__)
    _mm256_storeu_si256((__m256i *)a, x);
#elif defined(__SSE2__)
    _mm_storeu_si128((__m128i *)a, x);
#elif defined(__ALTIVEC__)
#else
    a[0] = x;
#endif
  }

  //
  // Non-member functions
  //
  inline friend int_vector operator*(int_vector a, const int_vector b) {
    a *= b;
    return a;
  }

  inline friend int_vector operator+(int_vector a, const int_vector b) {
    a += b;
    return a;
  }

  inline friend int_vector operator-(int_vector a, const int_vector b) {
    a -= b;
    return a;
  }

  inline friend int_vector operator&(int_vector a, const int_vector b) {
    a &= b;
    return a;
  }

  inline int sum() {
#if defined(__AVX__)
    int tmp[INT_VECTOR_LEN];
    this->copy(tmp);
    int res = 0;
    for (int i=0;i < INT_VECTOR_LEN;i++) res += tmp[i];
    return res;
#elif defined(__SSE2__)
    int tmp[INT_VECTOR_LEN];
    this->copy(tmp);
    int res = 0;
    for (int i=0;i < INT_VECTOR_LEN;i++) res += tmp[i];
    return res;
#elif defined(__ALTIVEC__)
    return 1;
#else
    return a;
#endif
  }

  inline friend int_vector gt_mask(const int_vector a, const int_vector b) {
#if defined(__AVX__)
    return int_vector(_mm256_cmpgt_epi32(a.x, b.x));
#elif defined(__SSE2__)
    return int_vector(_mm_cmpgt_epi32(a.x, b.x));
#elif defined(__ALTIVEC__)
    return a;
#else
    return int_vector(a.x > b.x);
#endif
  }

  inline friend int_vector lt_mask(const int_vector a, const int_vector b) {
    return gt_mask(b, a);
  }

  inline friend int_vector ge_mask(const int_vector a, const int_vector b) {
#if defined(__SSE2__)
    return( ~lt_mask(a, b) );
#elif defined(__ALTIVEC__)
    return a;
#else
    return (a.x >= b.x);
#endif
  }

  inline friend int_vector le_mask(const int_vector a, const int_vector b) {
    return ge_mask(b, a);
  }

  inline friend int_vector eq_mask(const int_vector a, const int_vector b) {
#if defined(__AVX__)
    return int_vector(_mm256_cmpeq_epi32(a.x, b.x));
#elif defined(__SSE2__)
    return int_vector(_mm_cmpeq_epi32(a.x, b.x));
#elif defined(__ALTIVEC__)
    return a;
#else
    return int_vector(a.x == b.x);
#endif
  }

  inline friend int_vector neq_mask(const int_vector a, const int_vector b) {
    return ~eq_mask(a, b);
  }

  inline friend int_vector mask_to_bool(const int_vector a) {
#if defined(__AVX__)
    return int_vector(_mm256_srli_epi32(a.x, 31));
#elif defined(__SSE2__)
    return int_vector(_mm_srli_epi32(a.x, 31));
#elif defined(__ALTIVEC__)
    return a;
#else
    return a;
#endif
  }

  inline friend int_vector operator==(const int_vector a, const int_vector b) {
    return mask_to_bool(eq_mask(a, b));
  }

  inline friend int_vector operator!=(const int_vector a, const int_vector b) {
    return mask_to_bool(neq_mask(a, b));
  }

  // Zero extended shift right
  inline friend int_vector zesr(const int_vector a, const int n) {
#if defined(__AVX__)
    return int_vector ( _mm256_srli_epi32(a.x, n) );
#elif defined(__SSE2__)
    return int_vector ( _mm_srli_epi32(a.x, n) );
#elif defined(__ALTIVEC__)
    return a;
#else
    return int_vector( ((unsigned int)a.x >> n) );
#endif
  }

  inline friend int_vector mulhi(const int_vector a, const int_vector b) {
#if defined(__AVX__)
    __m256i a1 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(a.x));
    __m256i b1 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(b.x));
    __m256i c1 = _mm256_srli_epi64(_mm256_mul_epi32(a1, b1), 32);

    __m256i a2 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128( _mm256_permute4x64_epi64(a.x, 0b1110) ));
    __m256i b2 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128( _mm256_permute4x64_epi64(b.x, 0b1110) ));
    __m256i c2 = _mm256_srli_epi64(_mm256_mul_epi32(a2, b2), 32);

    // Pack into lower 128 bits
    c1 = _mm256_permutevar8x32_epi32(c1, _mm256_set_epi32(7,7,7,7,6,4,2,0));

    // Pack into higher 128 bits
    c2 = _mm256_permutevar8x32_epi32(c2, _mm256_set_epi32(6,4,2,0,7,7,7,7));

    __m256i c = _mm256_or_si256(c1, c2);
#elif defined(__SSE2__)
    __m128i a1 = _mm_cvtepi32_epi64(a.x);
    __m128i b1 = _mm_cvtepi32_epi64(b.x);
    __m128i c1 = _mm_srli_epi64(_mm_mul_epi32(a1, b1), 32);

    __m128i a2 = _mm_cvtepi32_epi64(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(a.x), _mm_castsi128_ps(a.x))));
    __m128i b2 = _mm_cvtepi32_epi64(_mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(b.x), _mm_castsi128_ps(b.x))));
    __m128i c2 = _mm_srli_epi64(_mm_mul_epi32(a2, b2), 32);

    __m128i c = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(c1), _mm_castsi128_ps(c2), 0b10001000));
#elif defined(__ALTIVEC__)
    return a;
#else
    int c = (((unsigned long long)((long long)a * (long long)b)) >> 32);
#endif
    return int_vector(c);
  }

  inline friend int_vector divfast(const int_vector n, const int_vector d,
    const int_vector M, const int_vector s, const int_vector n_add_sign) {

    int_vector q = mulhi(M, n);
    q += n*n_add_sign;

    int_vector s_ge_0 = ge_mask(s, int_vector(0));
    
    q >>= (s & s_ge_0);
    q += (zesr(q, 31) & s_ge_0);

    return q;
  }

  //
  // Remainder using int_fastdiv
  //
  inline friend int_vector remfast(const int_vector n, const int_vector d,
    const int_vector M, const int_vector s, const int_vector n_add_sign) {

    // int quotient = n / divisor;
    int_vector quotient = divfast(n, d, M, s, n_add_sign);
    // int remainder = n - quotient * divisor;
    int_vector remainder = n - quotient * d;
    return remainder;
  }

};

#ifdef DEBUG_INT_VECTOR
//
// Helper functions
//
void print_int32(const int_vector a) {
  for (int i=0;i < INT_VECTOR_LEN;i++) {
    printf("%d ", a[i]);
  }
  printf("\n");
}
#endif

#endif // INT_VECTOR_H