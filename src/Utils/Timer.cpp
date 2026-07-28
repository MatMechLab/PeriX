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
//+++ Function: the timer class
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Utils/Timer.h"
#include "Utils/MessagePrinter.h"

Timer::Timer(){
    m_StartTimer=std::chrono::high_resolution_clock::now();
    m_EndTimer=std::chrono::high_resolution_clock::now();
    m_DurationSeconds=0.0;
}
void Timer::resetTimer(){
    m_StartTimer=std::chrono::high_resolution_clock::now();
    m_EndTimer=std::chrono::high_resolution_clock::now();
    m_DurationSeconds=0.0;
}

void Timer::startTimer(){
    m_StartTimer=std::chrono::high_resolution_clock::now();
    m_DurationSeconds=0.0;
}
void Timer::endTimer(){
    m_EndTimer=std::chrono::high_resolution_clock::now();
    m_DurationSeconds=std::chrono::duration_cast<std::chrono::microseconds>(m_EndTimer-m_StartTimer).count()/1.0e6;
}
void Timer::printElapseTime(const string &instr)const{
    char buff[13];
    snprintf(buff,13,"%12.5e",m_DurationSeconds);
    if (instr.empty()) {
        MessagePrinter::printNormalTxt("Elapsed time: "+string(buff)+" [s]");
    }
    else {
        MessagePrinter::printNormalTxt("Elapsed time: "+string(buff)+" [s], "+instr);
    }
}
void Timer::printElapseTime(const string &instr,const bool &flag)const{
    char buff[16];
    string str;
    snprintf(buff,16,"%14.5e",getDurationInSecond());
    str=buff;
    if(flag) MessagePrinter::printStars();
    MessagePrinter::printNormalTxt(instr+", elapsed time="+str+" [s]");
    MessagePrinter::printStars();
}
