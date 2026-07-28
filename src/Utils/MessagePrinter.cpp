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

#include "Utils/MessagePrinter.h"

#include <atomic>   // serialising the error exit
#include <chrono>
#include <mutex>    // keeping one message intact when threads report at once
#include <cstdio>   // fileno
#include <cstdlib>  // getenv
#include <thread>
#include <unistd.h> // isatty

namespace {
    // A message is assembled from several printf() calls (colour escape, head,
    // body, tail), so without a lock the reports of different OpenMP threads
    // interleave character-by-character into unreadable noise -- exactly what a
    // per-node failure inside the parallel bond loop produces. Recursive because
    // the public printers legitimately call one another.
    std::recursive_mutex &printMutex() {
        static std::recursive_mutex m;
        return m;
    }
    // Set once the process has begun terminating on an error, so the threads
    // that were already on their way into printErrorTxt do not repeat the same
    // report after the exit banner has been written.
    std::atomic<bool> &exitingFlag() {
        static std::atomic<bool> f{false};
        return f;
    }
}

namespace {
    // Colour escape codes are emitted ONLY when the standard output is an
    // interactive terminal. Under SLURM/nohup/background redirection or a pipe
    // (i.e. a captured log file), stdout is not a TTY, so PeriX falls back to
    // plain text -- no "\033[1;37m***" byte-noise polluting the log. Explicit
    // overrides follow the de-facto CLI conventions: NO_COLOR forces plain text
    // even on a terminal, CLICOLOR_FORCE forces colour even through a pipe (for
    // "... | tee run.log" or "less -R"). Evaluated once -- the nature of stdout
    // does not change during a run -- and the C++11 magic-static init is
    // thread-safe should a message ever be printed from an OpenMP region.
    bool colorsEnabled(){
        static const bool enabled = []{
            if(std::getenv("NO_COLOR")       != nullptr) return false;
            if(std::getenv("CLICOLOR_FORCE") != nullptr) return true;
            return isatty(fileno(stdout)) != 0;
        }();
        return enabled;
    }
    // Emit the "reset to default colour" escape only when colours are on.
    void resetColor(){
        if(colorsEnabled()) printf("\033[0m");
    }
}

MessagePrinter::MessagePrinter(){
}
//*****************************************
void MessagePrinter::exitPeriX(){
    // An error detected inside an OpenMP parallel region (a degenerate moment
    // matrix, a missing operator, ...) reaches this from EVERY worker thread at
    // once. Concurrent exit() runs the process-wide static destructors -- the
    // CUDA runtime's among them -- from several threads at the same time, which
    // interleaves the report into unreadable garbage and then crashes in
    // library teardown. Let exactly one thread report and terminate; park the
    // rest, because returning would let them keep running against a half
    // destroyed process.
    if (exitingFlag().exchange(true)) {
        for (;;) std::this_thread::sleep_for(std::chrono::hours(1));
    }
    printStars(MessageColor::RED);
    printTxt("PeriX exit due to some errors",MessageColor::RED);
    printStars(MessageColor::RED);
    std::fflush(stdout);
    // Report FAILURE: this is the error path, and a zero status here tells any
    // driving script or CI job that a diverged/aborted run succeeded.
    exit(EXIT_FAILURE);
}
//*****************************************
void MessagePrinter::setColor(const MessageColor &color){
    if(!colorsEnabled()) return;// plain-text mode (non-TTY / NO_COLOR): no escapes
    switch (color) {
        case MessageColor::WHITE:
            printf("\033[1;37m");// set color to white
            break;
        case MessageColor::RED:
            printf("\033[1;91m");// set color to bright red
            break;
        case MessageColor::BLUE:
            printf("\033[1;94m");// set color to bright blue
            break;
        case MessageColor::GREEN:
            printf("\033[1;32m");// set color to green
            break;
        case MessageColor::YELLOW:
            printf("\033[1;33m");// set color to yellow
            break;
        case MessageColor::MAGENTA:
            printf("\033[1;35m");// set color to megenta
            break;
        case MessageColor::CYAN:
            printf("\033[1;36m");// set color to cyan
            break;
        default:
            break;
    }
}
//*****************************************
void MessagePrinter::printStars(const MessageColor &color){
    const std::lock_guard<std::recursive_mutex> lk(printMutex());
    setColor(color);
    for(int i=0;i<m_Words;i++){
        printf("*");
    }
    printf("\n");
    resetColor();// recover color (no-op in plain-text mode)
}
//*****************************************
void MessagePrinter::printDashLine(const MessageColor &color){
    const std::lock_guard<std::recursive_mutex> lk(printMutex());
    setColor(color);
    printf("***");
    for(int i=0;i<m_Words-6;i++){
        printf("-");
    }
    printf("***\n");
    resetColor();// recover color (no-op in plain-text mode)
}
//*****************************************
void MessagePrinter::printSingleTxt(const string &str,const MessageColor &color){
    const std::lock_guard<std::recursive_mutex> lk(printMutex());
    setColor(color);
    printf("%s",str.c_str());
    setColor(MessageColor::WHITE);
}
//*****************************************
void MessagePrinter::printNewLine(){
    const std::lock_guard<std::recursive_mutex> lk(printMutex());
    printf("\n");
}
//*****************************************
void MessagePrinter::printTxt(const string &str,const MessageColor &color){
    const std::lock_guard<std::recursive_mutex> lk(printMutex());
    setColor(color);
    string _Head="*** ";
    string _End=" !!! ***";
    int i1,i2,i3;
    setColor(color);
    _Head="*** ";
    _End=" !!! ***";
    if(str.length()<=m_Words-_Head.length()-_End.length()){
        printf("%s",_Head.c_str());
        printf("%s",str.c_str());

        i1=static_cast<int>(_Head.size());
        i2=static_cast<int>(_End.size());
        i3=static_cast<int>(str.size());

        for(int i=0;i<m_Words-i1-i2-i3;i++){
            printf(" ");
        }
        printf("%s\n",_End.c_str());
    }
    else{
        string substr1,substr2;
        substr1=str.substr(0,m_Words-_Head.length()-_End.length());
        substr2=str.substr(m_Words-_Head.length()-_End.length());
        printf("%s",_Head.c_str());
        printf("%s",substr1.c_str());
        printf("%s\n",_End.c_str());

        printf("%s",_Head.c_str());
        printf("%s",substr2.c_str());
        i1=static_cast<int>(_Head.size());
        i2=static_cast<int>(_End.size());
        i3=static_cast<int>(substr2.size());
        for(int i=0;i<m_Words-i1-i2-i3;i++){
            printf(" ");
        }
        printf("%s\n",_End.c_str());
    }
    resetColor();// recover color (no-op in plain-text mode)
}
//**********************************************************
void MessagePrinter::printErrorTxt(const string &str,const bool &flag){
    const std::lock_guard<std::recursive_mutex> lk(printMutex());
    if (exitingFlag().load(std::memory_order_relaxed)) return;// already reported
    string HeadTxt;
    string EndTxt;
    vector<string> subvec;
    MessagePrinter printer;
    int i1,i2,i3;
    HeadTxt="*** Error: ";
    EndTxt=" !!! ***";
    subvec=printer.splitStr2Vec(HeadTxt,EndTxt,str);
    i1=static_cast<int>(HeadTxt.size());
    i2=static_cast<int>(EndTxt.size());
    i3=static_cast<int>(str.size());
    setColor(MessageColor::RED);
    if(flag){
        printStars(MessageColor::RED);
    }
    for(const auto &it:subvec){
        setColor(MessageColor::RED);
        printf("%s",HeadTxt.c_str());
        setColor(MessageColor::WHITE);
        printf("%s",it.c_str());
        i3=static_cast<int>(it.size());
        for(int i=0;i<m_Words-i1-i2-i3;i++){
            printf(" ");
        }
        setColor(MessageColor::RED);
        printf("%s\n",EndTxt.c_str());
    }
    if(flag){
        printStars(MessageColor::RED);
    }
}
//**********************************************************
void MessagePrinter::printWarningTxt(const string &str,const bool &flag){
    const std::lock_guard<std::recursive_mutex> lk(printMutex());
    string HeadTxt="*** Warning: ";
    string EndTxt=" !!! ***";
    vector<string> subvec;
    MessagePrinter printer;
    int i1,i2,i3;

    HeadTxt="*** Warning: ";
    EndTxt=" !!! ***";
    subvec=printer.splitStr2Vec(HeadTxt,EndTxt,str);
    i1=static_cast<int>(HeadTxt.size());
    i2=static_cast<int>(EndTxt.size());
    i3=static_cast<int>(str.size());
    setColor(MessageColor::YELLOW);
    if(flag){
        printStars(MessageColor::YELLOW);
    }
    for(const auto &it:subvec){
        setColor(MessageColor::YELLOW);
        printf("%s",HeadTxt.c_str());
        setColor(MessageColor::WHITE);
        printf("%s",it.c_str());
        i3=static_cast<int>(it.size());
        for(int i=0;i<m_Words-i1-i2-i3;i++){
            printf(" ");
        }
        setColor(MessageColor::YELLOW);
        printf("%s\n",EndTxt.c_str());
    }
    if(flag){
        printStars(MessageColor::YELLOW);
    }
}
//**********************************************************
void MessagePrinter::printWelcomeTxt(const string &str){
    const std::lock_guard<std::recursive_mutex> lk(printMutex());
    setColor(MessageColor::BLUE);
    string Head="*** ";
    string End =" ***";
    printf("%s",Head.c_str());
    printf("%s",str.c_str());

    int i1=static_cast<int>(Head.size());
    int i2=static_cast<int>(End.size());
    int i3=static_cast<int>(str.size());

    for(int i=0;i<m_Words-i1-i2-i3;i++){
        printf(" ");
    }
    printf("%s\n",End.c_str());
    resetColor();// recover color (no-op in plain-text mode)

}
//**********************************************************
void MessagePrinter::printNormalTxt(const string &str,const MessageColor &color){
    const std::lock_guard<std::recursive_mutex> lk(printMutex());
    setColor(color);
    string HeadTxt="*** ";
    string EndTxt =" ***";
    int i1=static_cast<int>(HeadTxt.size());
    int i2=static_cast<int>(EndTxt.size());
    int i3=static_cast<int>(str.size());

    vector<string> strvec;
    MessagePrinter printer;
    strvec=printer.splitStr2Vec(HeadTxt,EndTxt,str);
    for(const auto &it:strvec){
        setColor(color);
        printf("%s",HeadTxt.c_str());
        printf("%s",it.c_str());
        i3=static_cast<int>(it.size());
        for(int i=0;i<m_Words-i1-i2-i3;i++){
            printf(" ");
        }
        printf("%s\n",EndTxt.c_str());
    }
    resetColor();// recover color (no-op in plain-text mode)
}
//**********************************************************
vector<string> MessagePrinter::splitStr2Vec(const string &headtxt,const string &endtxt,string str){
    int i1=static_cast<int>(headtxt.size());
    int i2=static_cast<int>(endtxt.size());
    int i3=static_cast<int>(str.size());
    vector<string> strvec;
    strvec.clear();
    if(i3<=m_Words-i1-i2){
        strvec.push_back(str);
    }
    else{
        strvec.clear();
        string substr;
        substr.clear();
        int count=0;
        int nWords=m_Words-i1-i2;
        for(int i=0;i<static_cast<int>(str.length());i++){
            substr.push_back(str.at(i));
            count+=1;
            if(static_cast<int>(substr.size())==nWords){
                strvec.push_back(substr);
                substr.clear();
                if(count==static_cast<int>(str.length())){
                    break;
                }
            }
            else{
                if(count==static_cast<int>(str.length())){
                    strvec.push_back(substr);
                    break;
                }
            }
        }
    }
    return strvec;
}