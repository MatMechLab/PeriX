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

#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

using std::string;
using std::cout;
using std::endl;

/**
 * this class implement the timer for time elapse estimation
 */
class Timer{
public:
    /**
     * constructor
     */
    Timer();
    /**
     * start the time counting
     */
    void startTimer();
    /**
     * end the time counting
     */
    void endTimer();
    /**
     * reset the timer
     */
    void resetTimer();
    /**
     * print out the elapsed time
     * @param str string, default one is empty string
     */
    void printElapseTime(const string &str="")const;

    /**
     * print out the elapsed time
     * @param str string, default one is empty string
     * @param flag true for double star lines, false for single star line(bottom)
     */
    void printElapseTime(const string &str="",const bool &flag=true)const;

    //*****************************************************
    //*** general gettings
    //*****************************************************
    /**
     * get the duration time between start and end timer in second
     */
    [[nodiscard]] inline double getDurationInSecond()const{
        return std::chrono::duration_cast<std::chrono::microseconds>(m_EndTimer-m_StartTimer).count()/1.0e6;
    }
    /**
     * get the duration time between start and end timer in minute
     */
    [[nodiscard]] inline double getDurationInMinute()const{
        return std::chrono::duration_cast<std::chrono::milliseconds>(m_EndTimer-m_StartTimer).count()/1.0e3/60.0;
    }

private:
    std::chrono::high_resolution_clock::time_point m_StartTimer;/**< the start timer */
    std::chrono::high_resolution_clock::time_point m_EndTimer;/**< the end timer */
    double m_DurationSeconds;/**< duration time in seconds */
};