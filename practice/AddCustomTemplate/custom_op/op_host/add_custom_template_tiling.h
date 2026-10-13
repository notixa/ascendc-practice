
#include "register/tilingdata_base.h"
#include <cstdint>

namespace optiling {
BEGIN_TILING_DATA_DEF(AddCustomTemplateTilingData)
TILING_DATA_FIELD_DEF(uint32_t, size);
TILING_DATA_FIELD_DEF(uint32_t, tile_num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustomTemplate, AddCustomTemplateTilingData)
} // namespace optiling
