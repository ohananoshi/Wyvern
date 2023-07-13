 /*
WYVERN NUMERICAL METHODS

Author: Guilherme Arruda

Github: https://github.com/ohananoshi/C_Projects/tree/main/numerical_methods

Created in: 09/05/23

Last updated: 14/06/23

*/

//======================= HEARDERS ==========================

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <math.h>

//=================================================== DEFINES ===========================================

#define REDUCE_ERROR_FLAG 0

//========================================= MACRO FUNCTIONS ============================================



//============================================ DATATYPES ===============================================

typedef double (*x_function)(double x);
typedef double (*xy_function)(double x, double y);
typedef double (*multi_var_function)(double var_count, double* vars);

enum DATA_TYPES{
    INT_32_BITS,
    UNSIGNED_INT_32_BITS,
    DOUBLE
};

//============================================= FUNCTIONS =================================================

//========================================== ARRAY UTIL FUNCTIONS ==============================================

void* array_create(uint32_t numbers_count, uint8_t type, ...){

    va_list number;
    va_start(number, type);

    switch (type)
    {
    case INT_32_BITS:
        {   
            int32_t* output = (int32_t*)calloc(numbers_count, sizeof(int32_t));

            for(uint32_t i = 0; i < numbers_count; i++){
                output[i] = va_arg(number, int32_t);
            }

            va_end(number);

            return (int32_t*)output;

            break;
        }
    
    case UNSIGNED_INT_32_BITS:
        {   
            uint32_t* output = (uint32_t*)calloc(numbers_count, sizeof(uint32_t));

            for(uint32_t i = 0; i < numbers_count; i++){
            output[i] = va_arg(number, uint32_t);
            }

            va_end(number);

            return (uint32_t*)output;

            break;
        }
    
    case DOUBLE:
        {   
            double* output = (double*)calloc(numbers_count, sizeof(double));

            for(uint32_t i = 0; i < numbers_count; i++){
            output[i] = va_arg(number, double);
            }

            va_end(number);

            return (double*)output;

            break;
        }
    
    default:

        fprintf(stderr, "type not match.");
        return NULL;

        break;
    }
}

void array_print(uint32_t v_columns, bool type, ...){

    va_list array;
    va_start(array, type);

    if(type){

        double* v = va_arg(array, double*);

        for(uint32_t i = 0; i < v_columns; i++){
        printf("%f ", v[i]);
        }
    }else{
        int32_t* v = va_arg(array, int32_t*);

        for(uint32_t i = 0; i < v_columns; i++){
        printf("%d ", v[i]);
        }
    }

}

double* array_concat(uint16_t array_count, uint32_t* array_sizes, ...){

    if(array_count == 0){
        fprintf(stderr, "no arrays to concatenate.");
        return NULL;
    }

    double* output = (double*)calloc(array_int_unsigned_sum(array_sizes, 3), sizeof(double));

    uint32_t temp = 0;

    va_list arrays;
    va_start(arrays, array_sizes);

    for(uint16_t i = 0; i < array_count; i++){
        double* temp_array = (double*)calloc(array_sizes[i], sizeof(double));

        temp_array =  va_arg(arrays, double*);

        for(uint32_t j = 0; j < array_sizes[i]; j++){
            output[temp + j] = temp_array[j];
        }
        temp += array_sizes[i];
        free(temp_array);
    }

    va_end(arrays);

    return output;
}

//--------------------------------------- ARRAY ITERATE FUNCTIONS ---------------------------------------------------

double array_float_sum(double* v, uint32_t lenght){
    double sum = 0;
    for(uint32_t i = 0; i < lenght; i++){
        sum += v[i];
    }
    return sum;
}

int32_t array_int_signed_sum(int32_t* v, uint32_t length){
    int32_t sum = 0;
    for(uint32_t i = 0; i < length; i++){
        sum += v[i];
    }
    return sum;
}

uint32_t array_int_unsigned_sum(uint32_t* v, uint32_t length){
    uint32_t sum = 0;
    for(uint32_t i = 0; i < length; i++){
        sum += v[i];
    }
    return sum;
}

double* array_product(double* array_1, double* array_2, uint32_t length){

    double* output = (double*)calloc(length, sizeof(double));

    for(uint32_t i = 0; i < length; i++){
        output[i] = array_1[i]*array_2[i];
    }

    return output;
}

double* array_div(double* array_1, double* array_2, uint32_t length){

    double* output = (double*)calloc(length, sizeof(double));

    for(uint32_t i = 0; i < length; i++){
        output[i] = array_1[i]/array_2[i];
    }

    return output;
}

int32_t* array_round(double* array, uint32_t length){

    int32_t* output = (int32_t*)calloc(length, sizeof(int32_t));

    for(uint32_t i = 0; i < length; i++){
        output[i] = (int32_t)llround(array[i]);
    }

    return output;
}

double* array_truncate(double* array, uint32_t length){
    double* output = (double*)calloc(length, sizeof(double));

    for(uint32_t i = 0; i < length; i++){
        output[i] = trunc(array[i]);
    }

    return output;
}

int32_t* array_abs(int32_t* array, uint32_t length){
    int32_t* output = (int32_t*)calloc(length, sizeof(int32_t));

    for(uint32_t i = 0; i < length; i++){
        output[i] = labs(array[i]);
    }

    return output;
}

double* array_inverse(double* array, uint32_t length){
   double* output = (double*)calloc(length, sizeof(double));

    for(uint32_t i = 0; i < length; i++){
        output[i] = 1/(array[i]);
    }

    return output;
}

double* array_reduce_less(double* array, uint32_t length, double number, bool is_equal){

    uint32_t counter = 1;
    double* output = (double*)calloc(1, sizeof(double));

    if(is_equal){
        for(uint32_t i = 0; i < length; i++){
            if(array[i] <= number){
                output[counter] = array[i];
                counter++;
                output = (double*)realloc(output, 1);
            }
        }
    }else{
        for(uint32_t i = 0; i < length; i++){
            if(array[i] < number){
                output[counter] = array[i];
                counter++;
                output = (double*)realloc(output, 1);
            }
        }
    }

    if((REDUCE_ERROR_FLAG == 1) && (counter == 1)) fprintf(stderr, "There is no number less than : %f", number);

    output[0] = (double)(counter - 1);

    return output;
}

double* array_reduce_bigger(double* array, uint32_t length, double number, bool is_equal){

    uint32_t counter = 1;
    double* output = (double*)calloc(1, sizeof(double));

    if(is_equal){
        for(uint32_t i = 0; i < length; i++){
            if(array[i] >= number){
                output[counter] = array[i];
                counter++;
                output = (double*)realloc(output, 1);
            }
        }
    }else{
        for(uint32_t i = 0; i < length; i++){
            if(array[i] > number){
                output[counter] = array[i];
                counter++;
                output = (double*)realloc(output, 1);
            }
        }
    }

    if((REDUCE_ERROR_FLAG == 1) && (counter == 1)) fprintf(stderr, "There is no number bigger than : %f", number);

    output[0] = (double)(counter - 1);

    return output;
}

void* array_reduce_parity(void* array, uint32_t length, bool is_signed, bool parity){

    uint32_t counter = 1;

    if(is_signed){
        int32_t* temp_array = (int32_t*)array;
        int32_t* output = (int32_t*)calloc(1, sizeof(int32_t));

        for(uint32_t i = 0; i < length; i++){
            if((temp_array[i] % 2 == 0) && parity == 1){
                output[counter] = temp_array[i];
                counter++;
                output = (int32_t*)realloc(output, 1);
            }else if((temp_array[i] % 2 == 1) && parity == 0){
                output[counter] = temp_array[i];
                counter++;
                output = (int32_t*)realloc(output, 1);
            }
        }

        output[0] = (int32_t)(counter - 1);

        return (int32_t*)output;

    }else{
        uint32_t* temp_array = (uint32_t*)array;
        uint32_t* output = (uint32_t*)calloc(1, sizeof(uint32_t));

        for(uint32_t i = 0; i < length; i++){
            if((temp_array[i] % 2 == 0) && parity == 1){
                output[counter] = temp_array[i];
                counter++;
                output = (uint32_t*)realloc(output, 1);
            }else if((temp_array[i] % 2 == 1) && parity == 0){
                output[counter] = temp_array[i];
                counter++;
                output = (uint32_t*)realloc(output, 1);
            }
        }

        output[0] = counter - 1;

        return (uint32_t*)output;
    }

}

//====================================== MATRIX UTIL FUNCTIONS ============================================

double** matrix_create(uint32_t lines, uint32_t columns, ...){

    double** output = (double**)calloc(lines, sizeof(double*));

    va_list number;
    va_start(number, columns);

    for(uint32_t i = 0; i < lines; i++){
        output[i] = (double*)calloc(columns, sizeof(double));
        for(uint32_t j = 0; j < columns; j++){
            output[i][j] = va_arg(number, double);
        }
    }

    return output;
}

void matrix_print(double** matrix, uint32_t line_number, uint32_t column_number){
    for(uint32_t i = 0; i < line_number; i++){
        for(uint32_t j = 0; j < column_number; j++){
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

//====================================== LINEAR ALGEBRA FUNCTIONS =========================================

double* array_add(double* array_1, double* array_2, uint32_t length){

    double* output = (double*)calloc(length, sizeof(double));

    for(uint32_t i = 0; i < length; i++){

        output[i] = array_1[i] + array_2[i];
    }

    return output;
}

double* array_scalar_product(double* array, uint32_t length, double scalar){

    double* output = (double*)calloc(length, sizeof(double));

    for(uint32_t i = 0; i < length; i++){
        output[i] = array[i]*scalar;
    }

    return output;
}

double array_dot_product(double* array_1, double* array_2, uint32_t length){

    double output = array_float_sum(array_product(array_1, array_2,length),length);
    return output;
}

double** matrix_add(double** matrix_1, double** matrix_2, uint32_t matrix_lines, uint32_t matrix_columns){
    double** m = (double**)calloc(matrix_lines, sizeof(double*));

    for(uint32_t i = 0; i < matrix_lines; i++){
        m[i] = (double*)calloc(matrix_columns, sizeof(double));
        for(uint32_t j = 0; j < matrix_columns; j++){
            m[i][j] = matrix_1[i][j] + matrix_2[i][j];
        }
    }

    return m;
}

double** matrix_scalar_product(double** matrix, double scalar, uint32_t matrix_lines, uint32_t matrix_columns){
    double** m = (double**)calloc(matrix_lines, sizeof(double*));

    for(uint32_t i = 0; i < matrix_lines; i++){
        m[i] = (double*)calloc(matrix_columns, sizeof(double));
        for(uint32_t j = 0; j < matrix_columns; j++){
            m[i][j] = scalar*matrix[i][j];
        }
    }

    return m;
}

double** matrix_product(double** matrix_1, uint32_t matrix_1_lines, uint32_t matrix_1_columns, double** matrix_2, uint32_t matrix_2_lines, uint32_t matrix_2_columns){
    if(matrix_1_columns != matrix_2_lines) fprintf(stderr, "ERROR, sizes not match.");

    double** result = (double**)calloc(matrix_1_lines, sizeof(double*));
    double* temp_column = (double*)calloc(matrix_1_columns, sizeof(double));

    for(uint32_t i = 0; i < matrix_1_lines; i++){
        result[i] = (double*)calloc(matrix_1_columns, sizeof(double));
    }

    for(uint32_t i = 0; i < matrix_1_lines; i++){

        for(uint32_t j = 0; j < matrix_1_columns; j++){
            temp_column[j] = matrix_2[j][i];
        }

        for(uint32_t j = 0; j < matrix_1_lines; j++){
            result[j][i] = array_dot_product(matrix_1[j], temp_column, matrix_1_columns);
        }
    }

    free(temp_column);

    return result;
}

double** matrix_transpose(double** matrix, uint32_t lines, uint32_t columns){

    double** output = (double**)calloc(columns, sizeof(double*));

    for(uint32_t i = 0; i < columns; i++){

        output[i] = (double*)calloc(lines, sizeof(double));

        for(uint32_t j = 0; j < lines; j++){
            output[i][j] = matrix[j][i];
        }
    }

    return output;
}

void matrix_gaussian_elimination(double** matrix, uint32_t lines, uint32_t columns){

    double** output = (double**)calloc(lines, sizeof(double*));
    for(uint32_t i = 0; i < lines; i++){
        output[i] = (double*)calloc(columns, sizeof(double));
    }

    uint32_t pivot_counter = 0;
    double pivot;
    double* temp_1;
    double* temp_2;

    for(uint32_t i = 0; i < lines; i++){
        if(matrix[i][pivot_counter] == 0) continue;
        
        pivot = matrix[i][pivot_counter];
        temp_1 = matrix[i];

        array_scalar_product(temp_1, columns, 1/pivot);

        for(uint32_t j = i; j < lines; j++){
            output[j] = array_add(temp_1, matrix[j], columns);
        }
    }

    matrix_print(output, lines, columns);


}
/*test
double* matrix_gauss_elimination(double** matrix, uint32_t matrix_columns, uint32_t matrix_lines){

    double temp, pivot;
    double** result = (double**)calloc(matrix_columns, sizeof(double*));
    bool pivot_zero_flag = 0;

    matrix_print(matrix, matrix_lines, matrix_columns);

    for(uint32_t i = 0, k = 0; i < matrix_lines; i++){
        if(pivot_zero_flag) i = k;
        pivot = matrix[i][k];

        //printf("\n pivot: %f\n", pivot);

        if(pivot == 0){
            for(uint32_t j = k; j < matrix_columns; j++){
                temp = matrix[i][j];
                matrix[i][j] = matrix[i+1][j];
                matrix[i+1][j] = temp;
            }
            pivot_zero_flag = 1;
            matrix_print(matrix, matrix_lines, matrix_columns);
        }else{
            for(uint32_t u = k; u < matrix_columns; u++){
                matrix[i][u] /= pivot;
            }
            for(uint32_t u = k+1; u < matrix_lines; u++){
                temp = matrix[u][k];
                printf("temp: %f\n", matrix[k+1][k]);
                for(uint32_t v = k; v < matrix_columns; v++){
                    printf("%f = %f - (%f*%f) = ", matrix[u][v],matrix[u][v],temp,matrix[i][k]);
                    matrix[u][v] = matrix[u][v] - (temp*matrix[i][k]);
                    printf("%f\n", matrix[u][v]);
                }
            }
            matrix_print(matrix, matrix_lines, matrix_columns);
            pivot_zero_flag = 0;
            k++;
        }
    }

    matrix_print(matrix, matrix_lines, matrix_columns);

    return NULL;

}
*/
//========================================= RANGE FUNCTION ==============================================

void* range(uint8_t output_data_type, ...){

    switch (output_data_type){
        case INT_32_BITS:

            {
            int32_t start, end, step;

            va_list range_parameters;
            va_start(range_parameters, output_data_type);

            start = va_arg(range_parameters, int32_t);
            end = va_arg(range_parameters, int32_t);
            step = va_arg(range_parameters, int32_t);

            va_end(range_parameters);

            if((start < _I32_MIN) || (start > _I32_MAX)) fprintf(stderr, "start value not match");
            if((end < _I32_MIN) || (end > _I32_MAX)) fprintf(stderr, "end value not match");
            if((step < _I32_MIN) || (step > _I32_MAX) || (step > (abs(end-start)))) fprintf(stderr, "step size not match");

            int32_t array_lenght = round(fabs((end-start)/step) + 1);

            int32_t* temp_array = (int32_t*)calloc(array_lenght, sizeof(int32_t));

            for(int32_t i = 0; i < array_lenght; i++){
                temp_array[i] = start + i*step;
            }

            return temp_array;
            }

        case UNSIGNED_INT_32_BITS:

            {
            uint32_t start, end, step;

            va_list range_parameters;
            va_start(range_parameters, output_data_type);

            start = va_arg(range_parameters, uint32_t);
            end = va_arg(range_parameters, uint32_t);
            step = va_arg(range_parameters, uint32_t);

            va_end(range_parameters);

            int32_t array_lenght = round(fabs((end-start)/step) + 1);

            uint32_t* temp_array = (uint32_t*)calloc(array_lenght, sizeof(uint32_t));

            for(int32_t i = 0; i < array_lenght; i++){
                temp_array[i] = start + i*step;
            }

            return temp_array;
            }

        case DOUBLE:

            {
            double start, end, step;

            va_list range_parameters;
            va_start(range_parameters, output_data_type);

            start = va_arg(range_parameters, double);
            end = va_arg(range_parameters, double);
            step = va_arg(range_parameters, double);

            va_end(range_parameters);

            int32_t array_lenght = round(fabs((end-start)/step) + 1);

            double* temp_array = (double*)calloc(array_lenght, sizeof(double));

            for(int32_t i = 0; i < array_lenght; i++){
                temp_array[i] = start + i*step;
            }

            return temp_array;
            }

        default:

            fprintf(stderr, "type not match");

            return NULL;
    }

}

//============================ FUNCTIONS RANGE ==============================================

double* func_range(x_function function, double start_point, double end_point, double step){
    double array_lenght = round(fabs((end_point-start_point)/step) + 1);

    double* temp_array = (double*)calloc((uint32_t)array_lenght, sizeof(double));

    for(uint32_t i = 0; i < array_lenght; i++){
        temp_array[i] = function(start_point + i*step);
    }

    return temp_array;
}

double* func_input_x(x_function function, uint32_t input_array_size, double* input_array){

    double* out_array = (double*)calloc(input_array_size, sizeof(double));

    for(uint32_t i = 0; i < input_array_size; i++){
        out_array[i] = function(input_array[i]);
    }

    return out_array;
}

double* func_input_xy(xy_function function,uint32_t arrays_std_size, double* x_input_array, double* y_input_array){

    double* out_array = (double*)calloc(arrays_std_size, sizeof(double));

    for(uint32_t i = 0; i < arrays_std_size; i++){
        out_array[i] = function(x_input_array[i], y_input_array[i]);
    }

    return out_array;
}

double* func_input_multivar(multi_var_function function, uint32_t array_count, uint32_t array_std_size, ...){

    double* out_array = (double*)calloc(array_std_size, sizeof(double));
    double** temp_array = (double**)calloc(array_count, sizeof(double*));
    double* temp_array2 = (double*)calloc(array_count, sizeof(double));

    va_list arrays;
    va_start(arrays, array_std_size);

    for(uint32_t i = 0; i < array_count; i++){
        temp_array[i] = (double*)calloc(array_std_size, sizeof(double));
        temp_array[i] = va_arg(arrays, double*);
    }

    for(uint32_t i = 0; i < array_std_size; i++){
        for(uint32_t j = 0; j < array_count; j++){
            temp_array2[j] = temp_array[j][i];
        }

        out_array[i] = function(array_std_size, temp_array2);
    }

    return out_array;
}

//========================= FUNCTIONS ROOTS (1 VARIABLE) ======================================

double* root_interval_finder(x_function function, double start_point, double end_point, double step){
    uint32_t root_counter = 0;
    uint32_t j = 1;

    double* intervals = (double*)calloc(1, sizeof(double));
    double* range_array = func_range(function, start_point, end_point, step);
    uint32_t range_array_size = round(fabs(end_point - start_point)/step);


    for(uint32_t i = 1; i < range_array_size; i++){
        if((range_array[i-1] < 0 && range_array[i] > 0) || (range_array[i-1] > 0 && range_array[i] < 0)){
            root_counter++;
            intervals = (double*)realloc(intervals, (2*root_counter + 1)*sizeof(double));

            intervals[j] = start_point + (i-1)*step;
            intervals[j+1] = start_point + i*step;
            j += 2;
        } 
    }
    intervals[0] = root_counter;

    return intervals;
}

double root_bissec(x_function function, double interval_start, double interval_end, double precision){
    if(function(interval_start)*function(interval_end) > 0){
        fprintf(stderr, "No roots or many roots in the interval");
        return 0;
    }
    if((interval_end - interval_start) < precision ) return interval_start;

    double fp,
            x,
            start_buffer = interval_start,
            end_buffer = interval_end,
            result = (end_buffer - start_buffer);

    while(1){
        fp = function(start_buffer);
        x = (start_buffer + end_buffer)/2;

        if(fp*x > 0){
            start_buffer = x;
            if((end_buffer - start_buffer) < precision){
                result = start_buffer;
                break;
            }
        }
        else{
            end_buffer = x;
            if((end_buffer - start_buffer) < precision){
                result = end_buffer;
                break;
            }
        }
    }
    
    return result;
}

double root_regula_falsi(x_function function, double interval_start, double interval_end, double precision_1, double precision_2){
    if((interval_end - interval_start) < precision_1) return interval_start;
    if(fabs(function(interval_start)) < precision_2 || fabs(function(interval_end)) < precision_2) return interval_start;

    double fp,
            x,
            start_buffer = interval_start,
            end_buffer = interval_end,
            result = (end_buffer - start_buffer);

    while(1){
        fp = function(start_buffer);
        x = (start_buffer*function(end_buffer) - end_buffer*function(start_buffer))/(function(end_buffer) - function(start_buffer));

        if(fabs(function(x)) < precision_2){
            result = x;
            break;
        }
        if(fp*function(x) > 0){
            start_buffer = x;
            if((end_buffer - start_buffer) < precision_1){
                result = start_buffer;
                break;
            }
        }
        end_buffer = x;
        if((end_buffer - start_buffer) < precision_1){
            result = end_buffer - precision_1/2;
            break;
        }
    }

    return result;
}

//double root_sec(x_function function, double x_0, double x_1, double precision_1, double precision_2){
//   
//}


//================================ NUMERICAL INTEGRATION FUNCTIONS ============================================

//------------------------------------------- ONE VARIABLE ---------------------------------------------------

double integrate_composite_ret(x_function function, double start_point, double end_point, double step){
    double sum = 0;
    uint32_t width = round(fabs(end_point - start_point)/step);

    for(uint32_t i = 0; i < width; i++){
        sum += function(start_point + i*step);
    }

    return step*sum;
}

double integrate_composite_midpoint(x_function function, double start_point, double end_point, double step){
    double sum;
    uint32_t width = round(fabs(end_point - start_point)/step);

    for(uint32_t i = 0; i < width; i++){
        sum += function((2.00*start_point + step*(2.00*i + 1))/2.00);
    }

    return step*sum;
}

double integrate_composite_trapezoid(x_function function, double start_point, double end_point, double step){
    double sum = 0;
    uint32_t width = round(fabs(end_point - start_point)/step);

    for(uint32_t i = 0; i < width; i++){
        sum += function(start_point + i*step);
    }

    return step*(0.5*(function(start_point) + function(end_point)) + sum);
}

double integrate_simpson_1by3(x_function function, double start_point, double end_point){
    return (end_point - start_point)*(function(start_point) + 4.00*function((start_point + end_point)/2.00) + function(end_point))/6.00;
}

double integrate_simpson_3by8(x_function function, double start_point, double end_point){
    double h = (end_point - start_point)/3.00;
    return 3*h*(function(start_point) + 3.00*function(start_point + h) + 3.00*function(start_point + 2.00*h) + function(end_point))/8.00;
}

//double integrate_by_vector(double* function_values, uint32_t array_size){
//    
//}

//================================================== CURVE FITTING =========================================================

//------------------------------------------------ Linear Regression --------------------------------------------------------

double regression_linear(double* data[2], uint32_t data_columns, bool coefficient){
    double b1, b0, xi = 0, yi = 0, xi_yi = 0, xi_2 = 0;
    if(coefficient){
        for(uint32_t i = 0; i < data_columns; i++){
            xi += data[i][0];
            yi += data[i][1];
            xi_yi += data[i][0]*data[i][1];
            xi_2 += pow(data[i][0], 2.00);
        }

        b1 = ((xi*yi)-(data_columns*xi_yi))/(pow(xi, 2.00) - (data_columns*xi_2));
        return b1;
    }
    else{
        for(uint32_t i = 0; i < data_columns; i++){
            xi += data[i][0];
            yi += data[i][1];
            xi_yi += data[i][0]*data[i][1];
            xi_2 += pow(data[i][0], 2.00);
        }

        b1 = ((xi*yi)-(data_columns*xi_yi))/(pow(xi, 2.00) - (data_columns*xi_2));

        b0 = (yi - (b1*xi))/data_columns;

        return b0;
    }
}

double regression_linear_quality_adj(x_function reg_function, double* data[2], uint32_t data_columns, bool type){
    double a = 0, b = 0, c = 0;

    for(uint32_t i = 0; i < data_columns; i++){
        a += data[i][1];
        b += pow(data[i][i] - reg_function(data[i][0]), 2.00);
        c += pow(data[i][1], 2.00);
    }

    if(type) return (1 - (b/(c - (pow(a, 2.00)/data_columns))));
    else return (b - (data_columns - 2));
}

//------------------------------------------------- POLYNOMIAL REGRESSION -------------------------------------------------------
