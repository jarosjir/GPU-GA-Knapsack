/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef __Engine_dot_hpp_
#define __Engine_dot_hpp_

#include "../features/compilerfeatures.h"
#include "../array.h"
#include <limits>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <vector>
#include <stdint.h>

namespace r123{
/**
  If G satisfies the requirements of a CBRNG, and has a ctr_type whose
  value_type is an unsigned integral type, then Engine<G> satisfies
  the requirements of a C++0x "Uniform Random Number Engine" and can
  be used in any context where such an object is expected.

  Note that wrapping a counter based RNG with a traditional API in
  this way obscures much of the power of counter based PRNGs.
  Nevertheless, it may be of value in applications that are already
  coded to work with the C++0x random number engines.

  The MicroURNG template in MicroURNG.hpp
  provides the more limited functionality of a C++0x "Uniform
  Random Number Generator", but leaves the application in  control
  of counters and keys and hence may be preferable to the Engine template.
  For example, a MicroURNG allows one to use C++0x "Random Number
  Distributions"  without giving up control over the counters
  and keys.
*/ 

template<typename CBRNG>
struct Engine {
    typedef CBRNG cbrng_type;
    typedef typename CBRNG::ctr_type ctr_type;
    typedef typename CBRNG::key_type key_type;
    typedef typename CBRNG::ukey_type ukey_type;
    typedef typename ctr_type::value_type result_type;
    typedef size_t elem_type;

protected:
    cbrng_type b;
    key_type key;
    ukey_type ukey;
    ctr_type c;
    elem_type elem;
    ctr_type v;

    void fix_invariant(){
        if( elem != 0 ) {
            v = b(c, key);
	}
    }        
public:
    Engine() : b(), c(), elem() {
	ukey_type x = {{}};
	ukey = x;
        key = ukey;
    }
    Engine(result_type r) : b(), c(), elem() {
        std:: vector<uint32_t> ss(ukey.assembly_count());
        R123_ULONG_LONG rll = r;
        for(size_t i=0; rll && (i<ss.size()); ++i){
            ss[i] = uint32_t(rll);
            rll >>= 16; rll >>= 16; // dont warn, even if long long has only 32 bits.
        }        
        ukey.assemble(&ss[0]);
        key = ukey;
    }
    // Should one jump through template meta-programming hoops
    // here to force the compiler to resolve integral promotions
    // in favor of the result_type constructor rather than the
    // templated SeedSeq constructor??
    template <typename SeedSeq>
    Engine(SeedSeq s) : b(), c(), elem() {
        std::vector<uint32_t> ss(ukey.assembly_count());
        s.generate(ss.begin(), ss.end());
        ukey.assemble(&ss[0]);
        key = ukey;
    }

    template <typename SeedSeq>
    void seed(SeedSeq s){ 
        *this = Engine(s);
    }
    void seed(result_type r){
        *this = Engine(r);
    }
    void seed(){
        *this = Engine();
    }
    friend bool operator==(const Engine& lhs, const Engine& rhs){
        return lhs.c==rhs.c && lhs.elem == rhs.elem && lhs.ukey == rhs.ukey;
    }
    friend bool operator!=(const Engine& lhs, const Engine& rhs){
        return lhs.c!=rhs.c || lhs.elem != rhs.elem || lhs.ukey!=rhs.ukey;
    }

    friend std::ostream& operator<<(std::ostream& os, const Engine& be){
        return os << be.c << " " << be.ukey << " " << be.elem;
    }

    friend std::istream& operator>>(std::istream& is, Engine& be){
        is >> be.c >> be.ukey >> be.elem;
        be.key = be.ukey;
        be.fix_invariant();
        return is;
    }

    static result_type min() { return 0; }
    static result_type max() { return std::numeric_limits<result_type>::max(); }

    result_type operator()(){
        if( c.size() == 1 )     // short-circuit the scalar case.  Compilers aren't mind-readers.
            return b(c.incr(), key)[0];
        if( elem == 0 ){
            v = b(c.incr(), key);
            elem = c.size();
        }
        return v[--elem];
    }

    void discard(R123_ULONG_LONG skip){
        // don't forget:  elem counts down
        size_t nelem = c.size();
	size_t sub = skip % nelem;
        skip /= nelem;
	if (elem < sub) {
	    elem += nelem;
	    skip++;
	}
	elem -= sub;
        c.incr(skip);
        fix_invariant();
    }
         
    //--------------------------
    // Some bonus methods, not required for a Random Number
    // Engine

    // A constructor and seed() method for key_type seem useful
    // They're no more ambiguous than the templates over SeedSeq :-(.
    Engine(const key_type& k) : key(k), c(), elem(){
    }

    // Forward the e(counter) to the CBRNG we are templated
    // on, using the current value of the key.
    ctr_type operator()(const ctr_type& c) const{
        return b(c, key);
    }

    // allow it to be seeded with a ukey_type.
    void seed(const ukey_type& _k){
        ukey = _k;
        key = _k;
        fix_invariant();
    }        

    // Since you can seed *this with a key_type, it seems reasonable
    // to allow the caller to know what seed *this is using.
    ukey_type getseed() const{
        return ukey;
    }

    // Maybe the caller want's to know the details of
    // the internal state, e.g., so it can call a different
    // bijection with the same counter.
    std::pair<ctr_type, elem_type> getcounter() const {
        return make_pair(c,  elem);
    }

    // And the inverse.
    void setcounter(const ctr_type& _c, elem_type _elem){
        static const size_t nelem = c.size();
        if( elem > nelem )
            throw std::range_error("Engine::setcounter called  with elem out of range");
        c = _c;
        elem = _elem;
        fix_invariant();
    }
};
} // namespace r123

#endif
