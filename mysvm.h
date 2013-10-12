/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_ML_HPP__
#define __OPENCV_ML_HPP__

#include "opencv2/core/core.hpp"
#include <limits.h>


#include <map>
#include <string>
#include <iostream>

#define CV_VAR_NUMERICAL    0
#define CV_VAR_ORDERED      0
#define CV_VAR_CATEGORICAL  1

/* log(2*PI) */
#define CV_LOG2PI (1.8378770664093454835606594728112)

/* columns of <trainData> matrix are training samples */
#define CV_COL_SAMPLE 0

/* rows of <trainData> matrix are training samples */
#define CV_ROW_SAMPLE 1

#define CV_IS_ROW_SAMPLE(flags) ((flags) & CV_ROW_SAMPLE)

#define CV_TYPE_NAME_ML_SVM         "opencv-ml-svm"

namespace My {

    class CV_EXPORTS_W CvStatModel
    {
    public:
        CvStatModel();
        virtual ~CvStatModel();

        virtual void clear();

        CV_WRAP virtual void save( const char* filename, const char* name=0 ) const;
        CV_WRAP virtual void load( const char* filename, const char* name=0 );

        virtual void write( CvFileStorage* storage, const char* name ) const;
        virtual void read( CvFileStorage* storage, CvFileNode* node );

    protected:
        const char* default_model_name;
    };

    struct CV_EXPORTS_W_MAP CvParamGrid
    {
        // SVM params type
        enum { SVM_C=0, SVM_GAMMA=1, SVM_P=2, SVM_NU=3, SVM_COEF=4, SVM_DEGREE=5 };

        CvParamGrid()
        {
            min_val = max_val = step = 0;
        }

        CvParamGrid( double min_val, double max_val, double log_step );
        //CvParamGrid( int param_id );
        bool check() const;

        CV_PROP_RW double min_val;
        CV_PROP_RW double max_val;
        CV_PROP_RW double step;
    };

    inline CvParamGrid::CvParamGrid( double _min_val, double _max_val, double _log_step )
    {
        min_val = _min_val;
        max_val = _max_val;
        step = _log_step;
    }

    /****************************************************************************************\
     *                                   Support Vector Machines                              *
     \****************************************************************************************/

    // SVM training parameters
    struct CV_EXPORTS_W_MAP CvSVMParams
    {
        CvSVMParams();
        CvSVMParams( int svm_type, int kernel_type,
                double degree, double gamma, double coef0,
                double Cvalue, double nu, double p,
                CvMat* class_weights, CvTermCriteria term_crit );

        CV_PROP_RW int         svm_type;
        CV_PROP_RW int         kernel_type;
        CV_PROP_RW double      degree; // for poly
        CV_PROP_RW double      gamma;  // for poly/rbf/sigmoid
        CV_PROP_RW double      coef0;  // for poly/sigmoid

        CV_PROP_RW double      C;  // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
        CV_PROP_RW double      nu; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
        CV_PROP_RW double      p; // for CV_SVM_EPS_SVR
        CvMat*      class_weights; // for CV_SVM_C_SVC
        CV_PROP_RW CvTermCriteria term_crit; // termination criteria
    };


    struct CV_EXPORTS CvSVMKernel
    {
        typedef void (CvSVMKernel::*Calc)( int vec_count, int vec_size, const float** vecs,
                const float* another, float* results );
        CvSVMKernel();
        CvSVMKernel( const CvSVMParams* params, Calc _calc_func );
        virtual bool create( const CvSVMParams* params, Calc _calc_func );
        virtual ~CvSVMKernel();

        virtual void clear();
        virtual void calc( int vcount, int n, const float** vecs, const float* another, float* results );

        const CvSVMParams* params;
        Calc calc_func;

        virtual void calc_non_rbf_base( int vec_count, int vec_size, const float** vecs,
                const float* another, float* results,
                double alpha, double beta );

        virtual void calc_linear( int vec_count, int vec_size, const float** vecs,
                const float* another, float* results );
        virtual void calc_rbf( int vec_count, int vec_size, const float** vecs,
                const float* another, float* results );
        virtual void calc_poly( int vec_count, int vec_size, const float** vecs,
                const float* another, float* results );
        virtual void calc_sigmoid( int vec_count, int vec_size, const float** vecs,
                const float* another, float* results );
    };


    struct CvSVMKernelRow
    {
        CvSVMKernelRow* prev;
        CvSVMKernelRow* next;
        float* data;
    };


    struct CvSVMSolutionInfo
    {
        double obj;
        double rho;
        double upper_bound_p;
        double upper_bound_n;
        double r;   // for Solver_NU
    };

    class CvSVMSolver
    {
    public:
        typedef bool (CvSVMSolver::*SelectWorkingSet)( int& i, int& j );
        typedef float* (CvSVMSolver::*GetRow)( int i, float* row, float* dst, bool existed );
        typedef void (CvSVMSolver::*CalcRho)( double& rho, double& r );

        CvSVMSolver();

        CvSVMSolver( int count, int var_count, const float** samples, schar* y,
                int alpha_count, double* alpha, double Cp, double Cn,
                CvMemStorage* storage, CvSVMKernel* kernel, GetRow get_row,
                SelectWorkingSet select_working_set, CalcRho calc_rho );
        virtual bool create( int count, int var_count, const float** samples, schar* y,
                int alpha_count, double* alpha, double Cp, double Cn,
                CvMemStorage* storage, CvSVMKernel* kernel, GetRow get_row,
                SelectWorkingSet select_working_set, CalcRho calc_rho );
        virtual ~CvSVMSolver();

        virtual void clear();
        virtual bool solve_generic( CvSVMSolutionInfo& si );

        virtual bool solve_c_svc( int count, int var_count, const float** samples, schar* y,
                double Cp, double Cn, CvMemStorage* storage,
                CvSVMKernel* kernel, double* alpha, CvSVMSolutionInfo& si );
        virtual bool solve_nu_svc( int count, int var_count, const float** samples, schar* y,
                CvMemStorage* storage, CvSVMKernel* kernel,
                double* alpha, CvSVMSolutionInfo& si );
        virtual bool solve_one_class( int count, int var_count, const float** samples,
                CvMemStorage* storage, CvSVMKernel* kernel,
                double* alpha, CvSVMSolutionInfo& si );

        virtual bool solve_eps_svr( int count, int var_count, const float** samples, const float* y,
                CvMemStorage* storage, CvSVMKernel* kernel,
                double* alpha, CvSVMSolutionInfo& si );

        virtual bool solve_nu_svr( int count, int var_count, const float** samples, const float* y,
                CvMemStorage* storage, CvSVMKernel* kernel,
                double* alpha, CvSVMSolutionInfo& si );

        virtual float* get_row_base( int i, bool* _existed );
        virtual float* get_row( int i, float* dst );

        int sample_count;
        int var_count;
        int cache_size;
        int cache_line_size;
        const float** samples;
        const CvSVMParams* params;
        CvMemStorage* storage;
        CvSVMKernelRow lru_list;
        CvSVMKernelRow* rows;

        int alpha_count;

        double* G;
        double* alpha;

        // -1 - lower bound, 0 - free, 1 - upper bound
        schar* alpha_status;

        schar* y;
        double* b;
        float* buf[2];
        double eps;
        int max_iter;
        double C[2];  // C[0] == Cn, C[1] == Cp
        CvSVMKernel* kernel;

        SelectWorkingSet select_working_set_func;
        CalcRho calc_rho_func;
        GetRow get_row_func;

        virtual bool select_working_set( int& i, int& j );
        virtual bool select_working_set_nu_svm( int& i, int& j );
        virtual void calc_rho( double& rho, double& r );
        virtual void calc_rho_nu_svm( double& rho, double& r );

        virtual float* get_row_svc( int i, float* row, float* dst, bool existed );
        virtual float* get_row_one_class( int i, float* row, float* dst, bool existed );
        virtual float* get_row_svr( int i, float* row, float* dst, bool existed );
    };


    struct CvSVMDecisionFunc
    {
        double rho;
        int sv_count;
        double* alpha;
        int* sv_index;
    };


    // SVM model
    class CV_EXPORTS_W CvSVM : public CvStatModel
    {
    public:
        // SVM type
        enum { C_SVC=100, NU_SVC=101, ONE_CLASS=102, EPS_SVR=103, NU_SVR=104 };

        // SVM kernel type
        enum { LINEAR=0, POLY=1, RBF=2, SIGMOID=3 };

        // SVM params type
        enum { C=0, GAMMA=1, P=2, NU=3, COEF=4, DEGREE=5 };

        CV_WRAP CvSVM();
        virtual ~CvSVM();

        CvSVM( const CvMat* trainData, const CvMat* responses,
                const CvMat* varIdx=0, const CvMat* sampleIdx=0,
                CvSVMParams params=CvSVMParams() );

        virtual bool train( const CvMat* trainData, const CvMat* responses,
                const CvMat* varIdx=0, const CvMat* sampleIdx=0,
                CvSVMParams params=CvSVMParams() );

        virtual bool train_auto( const CvMat* trainData, const CvMat* responses,
                const CvMat* varIdx, const CvMat* sampleIdx, CvSVMParams params,
                int kfold = 10,
                CvParamGrid Cgrid      = get_default_grid(CvSVM::C),
                CvParamGrid gammaGrid  = get_default_grid(CvSVM::GAMMA),
                CvParamGrid pGrid      = get_default_grid(CvSVM::P),
                CvParamGrid nuGrid     = get_default_grid(CvSVM::NU),
                CvParamGrid coeffGrid  = get_default_grid(CvSVM::COEF),
                CvParamGrid degreeGrid = get_default_grid(CvSVM::DEGREE),
                bool balanced=false );

        virtual float predict( const CvMat* sample, bool returnDFVal=false ) const;
        virtual float predict( const CvMat* samples, CV_OUT CvMat* results ) const;

        CV_WRAP CvSVM( const cv::Mat& trainData, const cv::Mat& responses,
                const cv::Mat& varIdx=cv::Mat(), const cv::Mat& sampleIdx=cv::Mat(),
                CvSVMParams params=CvSVMParams() );

        CV_WRAP virtual bool train( const cv::Mat& trainData, const cv::Mat& responses,
                const cv::Mat& varIdx=cv::Mat(), const cv::Mat& sampleIdx=cv::Mat(),
                CvSVMParams params=CvSVMParams() );

        CV_WRAP virtual bool train_auto( const cv::Mat& trainData, const cv::Mat& responses,
                const cv::Mat& varIdx, const cv::Mat& sampleIdx, CvSVMParams params,
                int k_fold = 10,
                CvParamGrid Cgrid      = CvSVM::get_default_grid(CvSVM::C),
                CvParamGrid gammaGrid  = CvSVM::get_default_grid(CvSVM::GAMMA),
                CvParamGrid pGrid      = CvSVM::get_default_grid(CvSVM::P),
                CvParamGrid nuGrid     = CvSVM::get_default_grid(CvSVM::NU),
                CvParamGrid coeffGrid  = CvSVM::get_default_grid(CvSVM::COEF),
                CvParamGrid degreeGrid = CvSVM::get_default_grid(CvSVM::DEGREE),
                bool balanced=false);
        CV_WRAP virtual float predict( const cv::Mat& sample, bool returnDFVal=false ) const;
        CV_WRAP_AS(predict_all) void predict( cv::InputArray samples, cv::OutputArray results ) const;

        CV_WRAP virtual int get_support_vector_count() const;
        virtual const float* get_support_vector(int i) const;
        virtual CvSVMParams get_params() const { return params; };
        CV_WRAP virtual void clear();

        static CvParamGrid get_default_grid( int param_id );

        virtual void write( CvFileStorage* storage, const char* name ) const;
        virtual void read( CvFileStorage* storage, CvFileNode* node );
        CV_WRAP int get_var_count() const { return var_idx ? var_idx->cols : var_all; }

    protected:

        virtual bool set_params( const CvSVMParams& params );
        virtual bool train1( int sample_count, int var_count, const float** samples,
                const void* responses, double Cp, double Cn,
                CvMemStorage* _storage, double* alpha, double& rho );
        virtual bool do_train( int svm_type, int sample_count, int var_count, const float** samples,
                const CvMat* responses, CvMemStorage* _storage, double* alpha );
        virtual void create_kernel();
        virtual void create_solver();

        virtual float predict( const float* row_sample, int row_len, bool returnDFVal=false ) const;

        virtual void write_params( CvFileStorage* fs ) const;
        virtual void read_params( CvFileStorage* fs, CvFileNode* node );

        void optimize_linear_svm();

        CvSVMParams params;
        CvMat* class_labels;
        int var_all;
        float** sv;
        int sv_total;
        CvMat* var_idx;
        CvMat* class_weights;
        CvSVMDecisionFunc* decision_func;
        CvMemStorage* storage;

        CvSVMSolver* solver;
        CvSVMKernel* kernel;
    };
}

#endif
