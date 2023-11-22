#include "SYCLMath/GenVector/MathUtil.h"
#include "SYCLMath/Boost.h"
#include "SYCLMath/Vector4D.h"
#include <chrono>
#include <sycl/sycl.hpp>
#include <vector>

#ifdef SINGLE_PRECISION
using Scalar = float;
#else
using Scalar = double;
#endif

using LVector =
    ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<Scalar>>;
using Boost = ROOT::Experimental::Boost;

template <class T>
using Vector = std::vector<T>; // ROOT::RVec<T>;


/*
class Boost
{
public:
  enum ELorentzRotationMatrixIndex
  {
    kLXX = 0,
    kLXY = 1,
    kLXZ = 2,
    kLXT = 3,
    kLYX = 4,
    kLYY = 5,
    kLYZ = 6,
    kLYT = 7,
    kLZX = 8,
    kLZY = 9,
    kLZZ = 10,
    kLZT = 11,
    kLTX = 12,
    kLTY = 13,
    kLTZ = 14,
    kLTT = 15
  };

  enum EBoostMatrixIndex
  {
    kXX = 0,
    kXY = 1,
    kXZ = 2,
    kXT = 3,
    kYY = 4,
    kYZ = 5,
    kYT = 6,
    kZZ = 7,
    kZT = 8,
    kTT = 9
  };

  //    Default constructor (identity transformation)
  Boost() { SetIdentity(); }

  //   Construct given a three Scalars beta_x, beta_y, and beta_z
  Boost(Scalar beta_x, Scalar beta_y, Scalar beta_z)
  {
    SetComponents(beta_x, beta_y, beta_z);
  }

  void SetIdentity()
  {
    // set identity boost
    fM[kXX] = 1.0;
    fM[kXY] = 0.0;
    fM[kXZ] = 0.0;
    fM[kXT] = 0.0;
    fM[kYY] = 1.0;
    fM[kYZ] = 0.0;
    fM[kYT] = 0.0;
    fM[kZZ] = 1.0;
    fM[kZT] = 0.0;
    fM[kTT] = 1.0;
  }

  template <class Avector>
  explicit Boost(const Avector &beta) { SetComponents(beta); }

  
  //   Construct given a pair of pointers or iterators defining the
  //   beginning and end of an array of three Scalars to use as beta_x, _y, and _z
  
  template <class IT>
  Boost(IT begin, IT end) { SetComponents(begin, end); }

  
  //   copy constructor
  
  Boost(Boost const &b)
  {
    *this = b;
  }

  Boost &
  operator=(Boost const &rhs)
  {
    for (unsigned int i = 0; i < 10; ++i)
    {
      fM[i] = rhs.fM[i];
    }
    return *this;
  }

  
  //     Set components from a beta vector
  
  template <class Avector>
  void
  SetComponents(const Avector &beta)
  {
    SetComponents(beta.x(), beta.y(), beta.z());
  }

  
  //   Set given a pair of pointers or iterators defining the beginning and end of
  //   an array of three Scalars to use as beta_x,beta _y, and beta_z
  
  template <class IT>
  void SetComponents(IT begin, IT end)
  {
    IT a = begin;
    IT b = ++begin;
    IT c = ++begin;
    (void)end;
    assert(++begin == end);
    SetComponents(*a, *b, *c);
  }

  
  //   Get given a pair of pointers or iterators defining the beginning and end of
  //   an array of three Scalars into which to place beta_x, beta_y, and beta_z
   
  template <class IT>
  void GetComponents(IT begin, IT end) const
  {
    IT a = begin;
    IT b = ++begin;
    IT c = ++begin;
    (void)end;
    assert(++begin == end);
    GetComponents(*a, *b, *c);
  }

  
  //   Get given a pointer or an iterator defining the beginning of
  //   an array into which to place beta_x, beta_y, and beta_z
   
  template <class IT>
  void GetComponents(IT begin) const
  {
    double bx, by, bz = 0;
    GetComponents(bx, by, bz);
    *begin++ = bx;
    *begin++ = by;
    *begin = bz;
  }

  void SetComponents(Scalar bx, Scalar by, Scalar bz)
  {
    // set the boost beta as 3 components
    Scalar bp2 = bx * bx + by * by + bz * bz;
    if (bp2 >= 1)
    {
      // GenVector::Throw (
      //                         "Beta Vector supplied to set Boost represents speed >= c");
      //  SetIdentity();
      return;
    }
    Scalar gamma = 1.0 / ROOT::Experimental::mysqrt(1.0 - bp2);
    Scalar bgamma = gamma * gamma / (1.0 + gamma);
    fM[kXX] = 1.0 + bgamma * bx * bx;
    fM[kYY] = 1.0 + bgamma * by * by;
    fM[kZZ] = 1.0 + bgamma * bz * bz;
    fM[kXY] = bgamma * bx * by;
    fM[kXZ] = bgamma * bx * bz;
    fM[kYZ] = bgamma * by * bz;
    fM[kXT] = gamma * bx;
    fM[kYT] = gamma * by;
    fM[kZT] = gamma * bz;
    fM[kTT] = gamma;
  }

  void GetComponents (Scalar& bx, Scalar& by, Scalar& bz) const {
   // get beta of the boots as 3 components
   Scalar gaminv = 1.0/fM[kTT];
   bx = fM[kXT]*gaminv;
   by = fM[kYT]*gaminv;
   bz = fM[kZT]*gaminv;
}

  LVector
  operator()(LVector &v) const
  {
    // apply bosost to a PxPyPzE LorentzVector
    Scalar x = v.Px();
    Scalar y = v.Py();
    Scalar z = v.Pz();
    Scalar t = v.E();
    return LVector(fM[kXX] * x + fM[kXY] * y + fM[kXZ] * z + fM[kXT] * t, fM[kXY] * x + fM[kYY] * y + fM[kYZ] * z + fM[kYT] * t, fM[kXZ] * x + fM[kYZ] * y + fM[kZZ] * z + fM[kZT] * t, fM[kXT] * x + fM[kYT] * y + fM[kZT] * z + fM[kTT] * t);
  }

    template <class CoordSystem>
  ROOT::Experimental::LorentzVector<CoordSystem>
  operator() (const ROOT::Experimental::LorentzVector<CoordSystem> & v) const {
    LVector xyzt(v);
    LVector r_xyzt = operator()(xyzt);
    return ROOT::Experimental::LorentzVector<CoordSystem> ( r_xyzt );
  }

    template <class A4Vector>
  inline
  A4Vector operator* (const A4Vector & v) const
  {
    return operator()(v);
  }


  template <class Foreign4Vector>
  Foreign4Vector
  operator()(const Foreign4Vector &v) const
  {
    LVector xyzt(v);
    LVector r_xyzt = operator()(xyzt);
    return Foreign4Vector(r_xyzt.X(), r_xyzt.Y(), r_xyzt.Z(), r_xyzt.T());
  }

  
  //   Equality/inequality operators
   
  bool operator == (const Boost & rhs) const {
    for (unsigned int i=0; i < 10; ++i) {
      if( fM[i] != rhs.fM[i] )  return false;
    }
    return true;
  }
  bool operator != (const Boost & rhs) const {
    return ! operator==(rhs);
  }


  ROOT::Experimental::DisplacementVector3D< ROOT::Experimental::Cartesian3D<Scalar> >
BetaVector() const {
   // get boost beta vector
   Scalar gaminv = 1.0/fM[kTT];
   return ROOT::Experimental::DisplacementVector3D< ROOT::Experimental::Cartesian3D<Scalar> >
      ( fM[kXT]*gaminv, fM[kYT]*gaminv, fM[kZT]*gaminv );
}

void GetLorentzRotation (Scalar r[]) const {
   // get Lorentz rotation corresponding to this boost as an array of 16 values
   r[kLXX] = fM[kXX];  r[kLXY] = fM[kXY];  r[kLXZ] = fM[kXZ];  r[kLXT] = fM[kXT];
   r[kLYX] = fM[kXY];  r[kLYY] = fM[kYY];  r[kLYZ] = fM[kYZ];  r[kLYT] = fM[kYT];
   r[kLZX] = fM[kXZ];  r[kLZY] = fM[kYZ];  r[kLZZ] = fM[kZZ];  r[kLZT] = fM[kZT];
   r[kLTX] = fM[kXT];  r[kLTY] = fM[kYT];  r[kLTZ] = fM[kZT];  r[kLTT] = fM[kTT];
}

void Rectify() {
   // Assuming the representation of this is close to a true Lorentz Rotation,
   // but may have drifted due to round-off error from many operations,
   // this forms an "exact" orthosymplectic matrix for the Lorentz Rotation
   // again.

   if (fM[kTT] <= 0) {
      //GenVector::Throw (
      //                        "Attempt to rectify a boost with non-positive gamma");
      return;
   }
   ROOT::Experimental::DisplacementVector3D< ROOT::Experimental::Cartesian3D<Scalar> > beta ( fM[kXT], fM[kYT], fM[kZT] );
   beta /= fM[kTT];
   if ( beta.mag2() >= 1 ) {
      beta /= ( beta.R() * ( 1.0 + 1.0e-16 ) );
   }
   SetComponents ( beta );
}


void Invert() {
   // invert in place boost (modifying the object)
   fM[kXT] = -fM[kXT];
   fM[kYT] = -fM[kYT];
   fM[kZT] = -fM[kZT];
}

Boost Inverse() const {
   // return inverse of boost
   Boost tmp(*this);
   tmp.Invert();
   return tmp;
}


private:
  Scalar fM[10];
};
*/

LVector *ApplyBoost(LVector *lv, Boost bst, sycl::queue queue, const size_t N,
                    const size_t local_size)
{

  LVector *lvb = new LVector[N];

#ifdef ROOT_MEAS_TIMING
  auto start = std::chrono::system_clock::now();
#endif

  { // Start of scope, ensures data copied back to host
    // Create device buffers. The memory is managed by SYCL so we should NOT
    // access these buffers directly.
    auto execution_range = sycl::nd_range<1>{
        sycl::range<1>{((N + local_size - 1) / local_size) * local_size},
        sycl::range<1>{local_size}};

    sycl::buffer<LVector, 1> lv_sycl(lv, sycl::range<1>(N));
    sycl::buffer<LVector, 1> lvb_sycl(lvb, sycl::range<1>(N));
    sycl::buffer<Boost, 1> bst_sycl(&bst, sycl::range<1>(1));

    queue.submit([&](sycl::handler &cgh)
                 {
      // Get handles to SYCL buffers.
      sycl::accessor lv_acc{lv_sycl, cgh, sycl::range<1>(N), sycl::read_only};
      sycl::accessor lvb_acc{lvb_sycl, cgh, sycl::range<1>(N),
                             sycl::write_only};
      sycl::accessor bst_acc{bst_sycl, cgh, sycl::range<1>(1),
                             sycl::read_write};

      cgh.parallel_for(execution_range,
                       [=](sycl::nd_item<1> item) {
                         size_t id = item.get_global_id().get(0);
                         if (id < N) {
                          auto bst_loc = bst_acc[0];
                           lvb_acc[id] = bst_loc(lv_acc[id]);//bst(lv[id]);
                         }
                       }

      ); });
  } // end of scope, ensures data copied back to host
  queue.wait();

#ifdef ROOT_MEAS_TIMING
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() *
      1e-6;
  std::cout << "sycl time " << duration << " (s)" << std::endl;
#endif

  return lvb;
}

LVector *GenVectors(int n)
{

  LVector *vectors = new LVector[n];

  // generate n -4 momentum quantities
  for (int i = 0; i < n; ++i)
  {
    // fill vectors
    vectors[i] = {1., 1., 1., 1.};
  }

  return vectors;
}

bool print_if_false(const bool assertion, size_t i)
{
  if (!assertion)
  {
    std::cout << "Assertion failed at index " << i << std::endl;
  }
  return assertion;
}

int main(int argc, char **argv)
{

  std::string arg1 = argv[1];
  std::size_t pos;
  std::size_t N = std::stoi(arg1, &pos);
  size_t local_size = 128;

  LVector *lv = GenVectors(N);

  Boost bst(0.3, 0.4, 0.5);

  static sycl::queue queue{sycl::default_selector_v};

  std::cout << "sycl::queue check - selected device:\n"
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  LVector *lvb = ApplyBoost(lv, bst, queue, N, local_size);

  // for (size_t i=0; i<N; i++)
  //   assert(print_if_false((std::abs(masses[i] - 2.) <= 1e-5), i) );

  for (size_t i = 0; i < N; i++)
    std::cout << lv[i] << " " << lvb[i] << std::endl;

  delete[] lv;
  delete[] lvb;
  return 0;
}
