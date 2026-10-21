#ifndef TILE_ELEMWISE_LEAKYRELU_HPP
#define TILE_ELEMWISE_LEAKYRELU_HPP
#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {
template <class ArchTag_, class ComputeType_, uint32_t COMPUTE_LENGTH_, uint32_t PERCENT = 1>
struct TileElemWiseLeakyRelu {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;
    static constexpr float SLOPE = PERCENT / 100.0f;

    CATLASS_DEVICE
    TileElemWiseLeakyRelu()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementCompute> const& dstLocal, AscendC::LocalTensor<ElementCompute> const& srcLocal)
    {
        using namespace AscendC;
        LeakyRelu(dstLocal, srcLocal, SLOPE, COMPUTE_LENGTH);
    }
};
} // namespace Catlass::Epilogue::Tile
#endif
