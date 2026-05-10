#include "axonforge/backend.hpp"
#include <mutex>
#include <stdexcept>

namespace axonforge {

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry registry;
    return registry;
}

void BackendRegistry::register_backend(std::string_view id, FactoryFn factory) {
    factories_.insert_or_assign(std::string(id), std::move(factory));
}

std::unique_ptr<IBackend> BackendRegistry::create(std::string_view id) const {
    auto it = factories_.find(std::string(id));
    if (it == factories_.end()) {
        throw std::runtime_error(
            "BackendRegistry: no backend registered with id '" + std::string(id) + "'");
    }
    return it->second();
}

bool BackendRegistry::has_backend(std::string_view id) const noexcept {
    return factories_.count(std::string(id)) > 0;
}

std::vector<std::string> BackendRegistry::available_backends() const {
    std::vector<std::string> ids;
    ids.reserve(factories_.size());
    for (const auto& [k, _] : factories_) ids.push_back(k);
    return ids;
}

} // namespace axonforge
