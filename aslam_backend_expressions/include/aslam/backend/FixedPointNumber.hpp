#ifndef ASLAM_BACKEND_FIXED_POINT_NUMBER_HPP
#define ASLAM_BACKEND_FIXED_POINT_NUMBER_HPP

#include <cstdint>
#include <limits>
#include <type_traits>
#include <iosfwd>

#if __cplusplus >= 201402L
#define AB_DEPRECATED(msg) [[deprecated(msg)]]
#else
#define AB_DEPRECATED(msg)
#endif

namespace aslam {
namespace backend {

namespace internal {
template <bool Cond, typename IfTrue, typename IfFalse>
struct SwitchType;

template <typename IfTrue, typename IfFalse>
struct SwitchType<true, IfTrue, IfFalse>{
  typedef IfTrue type;
};

template <typename IfTrue, typename IfFalse>
struct SwitchType<false, IfTrue, IfFalse>{
  typedef IfFalse type;
};

template <unsigned char SizeInBytes>
struct NextBiggerNumberTypeBySize;

template <>
struct NextBiggerNumberTypeBySize<1> {
  typedef std::int16_t type;
};

template <>
struct NextBiggerNumberTypeBySize<2> {
  typedef std::int32_t type;
};

template <>
struct NextBiggerNumberTypeBySize<4> {
  typedef std::int64_t type;
};

template <>
struct NextBiggerNumberTypeBySize<8> {
  typedef SwitchType<(sizeof(long long) > 8), long long, long double>::type type;
};

} // namespace internal

template <typename Integer_, std::uintmax_t Divider>
class FixedPointNumber{
 public:
  typedef Integer_ Integer;
  typedef typename internal::NextBiggerNumberTypeBySize<sizeof(Integer)>::type NextBiggerType;

  constexpr static std::uintmax_t getDivider(){ return Divider; }

  struct Numerator {
    Integer i;
    constexpr Numerator(Integer i) : i(i) {}
  };

  FixedPointNumber() {
    static_assert(std::numeric_limits<Integer_>::is_integer, "only integral types are allowed as Integer_");
    static_assert(Divider != 0, "the Divider must be zero");
    static_assert((1.0/(double)Divider) * Divider == 1, "the Divider must be loss less convertible to double");
    static_assert((std::uintmax_t)((Integer)Divider) == Divider, "the Divider must be loss less convertible to the Integer_ type");
  }

  FixedPointNumber(const FixedPointNumber & other) = default;

  AB_DEPRECATED("Use FixedPointNumber(Numerator num) instead of FixedPointNumber(Integer num). E.g. let FixedPointNumber<...>::Numerator(num) be automatically converted to FixedPointNumber<...> or use FixedPointNumber<...>::fromNumerator(num).")
  constexpr FixedPointNumber(Integer num) : _num(num){}

  constexpr FixedPointNumber(Numerator num) : _num(num.i){}
  constexpr static FixedPointNumber fromNumerator(Integer num) { return FixedPointNumber(Numerator(num)); }
  constexpr FixedPointNumber(double const & other) : _num(other * getDivider()) {}

  template <typename OtherInteger_, std::uintmax_t OtherDivider_>
  explicit FixedPointNumber(FixedPointNumber<OtherInteger_, OtherDivider_> const & other) { _num = other._num * getDivider() / other.getDivider(); }

  operator double() const { return (long double) _num / getDivider(); }
//  operator Integer() = delete; TODO find out : why does this disable conversions to double?
  Integer getNumerator() const { return _num; }
  operator Integer() const { return getNumerator(); }
  Integer getDenominator() const { return Integer(getDivider()); }

  FixedPointNumber operator - () const {
    return FixedPointNumber(Numerator(-_num));
  }

  FixedPointNumber operator + (const FixedPointNumber & other) const {
    return FixedPointNumber(Numerator(_num + other._num));
  }
  FixedPointNumber & operator += (const FixedPointNumber & other) {
    _num += other._num;
    return *this;
  }

  FixedPointNumber operator - (const FixedPointNumber & other) const {
    return FixedPointNumber(Numerator(_num - other._num));
  }
  FixedPointNumber & operator -= (const FixedPointNumber & other) {
    _num -= other._num;
    return *this;
  }

  FixedPointNumber operator * (const FixedPointNumber & other) const {
    return FixedPointNumber(Numerator(((NextBiggerType)_num * (NextBiggerType)other._num) / (NextBiggerType)other.getDivider()));
  }
  FixedPointNumber & operator *= (const FixedPointNumber & other) {
    *this = *this * other;
    return *this;
  }

  FixedPointNumber operator / (const FixedPointNumber & other) const {
    return FixedPointNumber(Numerator(((NextBiggerType)_num * (NextBiggerType)other.getDivider()) / (NextBiggerType)other._num));
  }
  FixedPointNumber & operator /= (const FixedPointNumber & other) {
    *this = *this / other;
    return *this;
  }

  bool operator == (const FixedPointNumber & other) const {
    return _num == other._num;
  }
  bool operator != (const FixedPointNumber & other) const {
    return _num != other._num;
  }
  bool operator < (const FixedPointNumber & other) const {
    return _num < other._num;
  }
  bool operator > (const FixedPointNumber & other) const {
    return _num > other._num;
  }
  bool operator <= (const FixedPointNumber & other) const {
    return _num <= other._num;
  }
  bool operator >= (const FixedPointNumber & other) const {
    return _num >= other._num;
  }

  friend std::ostream & operator << (std::ostream & o, const FixedPointNumber & v){
    o << typename std::enable_if<(sizeof(v) > 0), char>::type('[') << v.getNumerator() << " / " << v.getDenominator() << ']';
    return o;
  }

  constexpr static FixedPointNumber Zero() { return Numerator(0); }
 private:
  Integer _num;
  template<typename OtherInteger_, std::uintmax_t OtherDivider>
  friend class FixedPointNumber;
};

template <typename T>
struct is_fixed_point_number {
  constexpr static bool value = false;
};

template <typename Integer_, std::uintmax_t Divider>
struct is_fixed_point_number<FixedPointNumber<Integer_, Divider>> {
  constexpr static bool value = true;
};

}  // namespace backend
}  // namespace aslam

namespace std {

template <typename Integer_, std::uintmax_t Divider>
struct numeric_limits<aslam::backend::FixedPointNumber<Integer_, Divider> > {
  constexpr static double epsilon() { return 1.0 / Divider; }
  constexpr static bool is_integer = Divider == 1;
  constexpr static bool is_signed = std::numeric_limits<Integer_>::is_signed;
};
}

#endif /* ASLAM_BACKEND_FIXED_POINT_NUMBER_HPP */
