//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* All rights reserved, Yang Bai/MM-Lab@CopyRight 2026-present
//* https://github.com/MatMechLab/PeriX
//* Licensed under GNU GPLv3, please see LICENSE for details
//* https://www.gnu.org/licenses/gpl-3.0.en.html
//****************************************************************
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Author  : Yang Bai
//+++ Date    : 2026.04.14
//+++ Function: Implement a general json file reading/element access
//+++           for material properties and other calculations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Utils/JsonUtils.h"
#include <unordered_set>

JsonUtils::JsonUtils(){}

void JsonUtils::checkValidation(const nlohmann::json &t_json,const string &matename){
    if(!t_json.contains(matename)){
        MessagePrinter::printErrorTxt("can\'t find material property("+matename+") in the given json file, please check your input file");
        MessagePrinter::exitPeriX();
    }
}
double JsonUtils::getValue(const nlohmann::json &t_json,const string &matename){
    if(t_json.contains(matename)){
        const auto& value = t_json.at(matename);
        if(value.is_number()||
           value.is_number_float()||
           value.is_number_integer()||
           value.is_number_unsigned()){
            return static_cast<double>(value);
        }
        else{
            MessagePrinter::printErrorTxt("the value of material property(\'"+matename+"\') is not a valid number, please check your input file");
            MessagePrinter::exitPeriX();
        }
    }
    else{
        MessagePrinter::printErrorTxt("can\'t find material property(\'"+matename+"\') in the given json file, please check your input file");
        MessagePrinter::exitPeriX();
    }
    return 0.0;
}

int JsonUtils::getInteger(const nlohmann::json &t_json,const string &matename){
    if(t_json.contains(matename)){
        const auto& value = t_json.at(matename);
        if(value.is_number_integer()||
           value.is_number_unsigned()){
            return static_cast<int>(value);
        }
        else{
            MessagePrinter::printErrorTxt("the value of material property(\'"+matename+"\') is not a valid integer, please check your input file");
            MessagePrinter::exitPeriX();
        }
    }
    else{
        MessagePrinter::printErrorTxt("can\'t find material property(\'"+matename+"\') in the given json file, please check your input file");
        MessagePrinter::exitPeriX();
    }
    return -1;
}

string JsonUtils::getString(const nlohmann::json &t_json,const string &matename){
    if(t_json.contains(matename)){
        if(t_json.at(matename).is_string()){
            return t_json.at(matename);
        }
        else{
            MessagePrinter::printErrorTxt("the value of material property(\'"+matename+"\') is not a valid string, please check your input file");
            MessagePrinter::exitPeriX();
        }
    }
    else{
        MessagePrinter::printErrorTxt("can\'t find material property(\'"+matename+"\') in the given json file, please check your input file");
        MessagePrinter::exitPeriX();
    }
    return "";
}

bool JsonUtils::getBoolean(const nlohmann::json &t_json,const string &matename){
    if(t_json.contains(matename)){
        if(t_json.at(matename).is_boolean()){
            return t_json.at(matename);
        }
        else{
            MessagePrinter::printErrorTxt("the value of material property(\'"+matename+"\') is not a valid boolean, please check your input file");
            MessagePrinter::exitPeriX();
        }
    }
    else{
        MessagePrinter::printErrorTxt("can\'t find material property(\'"+matename+"\') in the given json file, please check your input file");
        MessagePrinter::exitPeriX();
    }
    return false;
}

Vector3d JsonUtils::getVector(const nlohmann::json &t_json,const string &matename){
    if(t_json.contains(matename)){
        if(!t_json.at(matename).is_array()){
            MessagePrinter::printErrorTxt("the vector value of material property(\'"+matename+"\') is not a valid vector(3d), please check your input file");
            MessagePrinter::exitPeriX();
        }
        else{
            Vector3d temp(0.0);
            for(int i=0;i<static_cast<int>(t_json.at(matename).size());i++){
                if(t_json.at(matename).at(i).is_number()){
                    temp(i+1)=static_cast<double>(t_json.at(matename).at(i));
                }
                else{
                    MessagePrinter::printErrorTxt(to_string(i+1)+"-th value of material property(\'"+matename+"\') is not a valid number, please check your input file");
                    MessagePrinter::exitPeriX();
                }
            }
            return temp;
        }
    }
    else{
        MessagePrinter::printErrorTxt("can\'t find material property(\'"+matename+"\') in the given json file, please check your input file");
        MessagePrinter::exitPeriX();
    }
    return Vector3d(0);
}

bool JsonUtils::hasValue(const nlohmann::json &t_json,const string &matename){
    if(t_json.contains(matename)){
        return true;
    }
    return false;
}


bool JsonUtils::hasOnlyGivenValues(const nlohmann::json &t_json,const vector<string> &namevec){
    const std::unordered_set<string> valid_names(namevec.begin(), namevec.end());
    for (const auto& [key, _] : t_json.items()) {
        if (valid_names.find(key) == valid_names.end()) {
            return false;
        }
    }
    return true;
}