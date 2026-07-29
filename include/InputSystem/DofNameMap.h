//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* Licensed under GNU GPLv3, please see LICENSE for details
//****************************************************************

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

/**
 * Ordered mapping from the public JSON DoF names to PeriX's one-based slots.
 *
 * The manuscript-facing input uses names exclusively. Numeric aliases are not
 * part of the public schema because they make a deck dependent on an implicit
 * field ordering.
 */
class DofNameMap {
public:
    void setNames(const std::vector<std::string> &names) {
        m_Names=names;
        m_Index.clear();
        for (std::size_t i=0;i<names.size();++i) {
            m_Index.emplace(names[i],static_cast<int>(i)+1);
        }
    }

    [[nodiscard]] bool empty() const { return m_Names.empty(); }
    [[nodiscard]] int size() const { return static_cast<int>(m_Names.size()); }
    [[nodiscard]] const std::vector<std::string>& names() const { return m_Names; }

    [[nodiscard]] int indexOf(const std::string &name) const {
        const auto it=m_Index.find(name);
        return it==m_Index.end() ? -1 : it->second;
    }

private:
    std::vector<std::string> m_Names;
    std::unordered_map<std::string,int> m_Index;
};
