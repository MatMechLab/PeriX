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
//+++ Function: implement the general message print using in
//+++           the terminal
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "Utils/MessageColor.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::to_string;


/**
 * this class offers general message output in the command-line
 */
class MessagePrinter{
public:
    /**
     * constructor
     */
    MessagePrinter();

    /**
     * print out the single txt or sentences
     * @param str the string to be printed
     * @param color the color for current string printing
    */
    static void printSingleTxt(const string &str,const MessageColor &color=MessageColor::WHITE);

    /**
     * print out new line
    */
    static void printNewLine();

    /**
     * print out the standard text
     * @param str the string to be printed
     * @param color the color for current string printing
     */
    static void printTxt(const string &str,const MessageColor &color=MessageColor::WHITE);


    /**
     * print out the text as an error message
     * @param str the string to be printed
     * @param flag true->print out the star in red color, otherwise in white color
     */
    static void printErrorTxt(const string &str,const bool &flag=true);
    /**
     * print out the text as a warning message
     * @param str the string to be printed
     * @param flag true->print out the star in yellow color, otherwise in white color
     */
    static void printWarningTxt(const string &str,const bool &flag=true);
    /**
     * print out a welcome text
     * @param str the string to be printed
     */
    static void printWelcomeTxt(const string &str);
    /**
     * print out a normal text string
     * @param str the string to be printed
     * @param color the color for current string printing
     */
    static void printNormalTxt(const string &str,const MessageColor &color=MessageColor::WHITE);


    /**
     * print out a star line
     * @param color the color for starts, default is white
     */
    static void printStars(const MessageColor &color=MessageColor::WHITE);
    /**
     * print out a dashed line
     * @param color the color for the dashed line, default is white
     */
    static void printDashLine(const MessageColor &color=MessageColor::WHITE);

    /**
     * this will stop all the process and exit the perix program!
     */
    static void exitPeriX();


    /**
     * set the color which will be used in your terminal output
     * @param color the color for terminal output
     */
    static void setColor(const MessageColor &color);

private:
    static const int m_Words=105;/**< the max number of characters in one single line output */
    /**
     * this function split the str into several substring based on the given head and end txt
     * @param headtxt head string txt
     * @param endtxt end string txt
     * @param str the input string
     */
    vector<string> splitStr2Vec(const string &headtxt,const string &endtxt,string str);
};