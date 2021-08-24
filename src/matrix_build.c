#include <stdio.h>
#include <stdlib.h>
// #include <stdbool.h>

#ifndef G
#define G 12
#endif

#ifndef I
#define I 100
#endif

#ifndef materials
#define materials 1
#endif

// Define 2D Array sizes
typedef struct full_matrix {
    double array[G*(I+1)][G*(I+1)];
} full_matrix;

typedef struct cross_section {
    double array[G][G];
} cross_section;

typedef struct boundary {
    double array[G][2];
} boundary;

// Multiple instances of G vectors
typedef struct multi_vec {
    double array[materials][G];
} multi_vec;

typedef struct multi_mat {
    double array[materials][G][G];
} multi_mat;

int change_space(int gg, int ii){
    return gg * (I + 1) + ii;
}

extern void construct_A_lambda(full_matrix *A, cross_section *scatter, boundary *BC, void *diff, void *surface, \
                            void *volume, void * remove, double delta) {

    double * D = (double *) diff;
    double * SA = (double *) surface;
    double * V = (double *) volume;
    double * removal = (double *) remove;

    int cell, prime;

    for (int gg = 0; gg < G; gg++){
        for (int ii = 0; ii < I; ii++){
            cell = change_space(gg,ii);
            A->array[cell][cell] = (2.0 / (delta * V[ii]) * \
                ((D[gg]*D[gg])/(D[gg] + D[gg]) * SA[ii+1]) + removal[gg]);
            A->array[cell][cell+1] = -2.0*(D[gg] * D[gg]) / (D[gg] + D[gg]) \
                 / (delta * V[ii]) * SA[ii+1];
            if (ii > 0){
                A->array[cell][cell-1] = -2.0*(D[gg] * D[gg]) / (D[gg] + D[gg]) \
                     / (delta * V[ii]) * SA[ii];
                A->array[cell][cell] += 2.0 / (delta * V[ii]) * ((D[gg]*D[gg])/ \
                     (D[gg] + D[gg]) * SA[ii]);
            }
            for (int gpr = 0; gpr < G; gpr++){
                if (gpr != gg){
                    prime = change_space(gpr,ii);
                    A->array[cell][prime] = -scatter->array[gg][gpr];
                }
            }
        }
        // Boundary Conditions
        cell = change_space(gg,I);
        A->array[cell][cell] = BC->array[gg][0]*0.5 + BC->array[gg][1] / delta;
        A->array[cell][cell-1] = BC->array[gg][0]*0.5 - BC->array[gg][1] / delta;
    }
}

extern void construct_A_list(full_matrix *A, multi_mat *scatter, boundary *BC, multi_vec *D, void *surface, \
                            void *volume, multi_vec * removal, void *shape, double delta) {

    double * SA = (double *) surface;
    double * V = (double *) volume;
    int * layers = (int *) shape;

    int first [materials] = {0};
    int last [materials] = {0};

    int total_sum = 0;
    for (int mat = 0; mat < materials; mat++){
        total_sum += layers[mat];
        first[mat] = total_sum;
        last[mat] = total_sum - 1;
    }
    
    int global_cell, prime;
    int interest, add = 0, sub = 0;
    int material_index = 0, material_cell = 0;

    for (int gg = 0; gg < G; gg++){
        material_index = 0;
        for (int local_cell = 0; local_cell < I; local_cell++){
            // This is for the material index
            if (local_cell == 0){
                material_cell = 0;
                material_index = 0;
            } 
            else if ((material_cell % layers[material_index]) == (layers[material_index] - 1)){
                material_cell = 0;          
                material_index += 1;
            } 
            else {
                material_cell += 1;
            }


            if ((local_cell == last[material_index]) && (local_cell != (I - 1))){
                add = 1;
            }
            else {
                add = 0;
            }

            if (local_cell == first[material_index - 1]){
                sub = 1;
            }
            else {
                sub = 0;
            }

            /*
            // For cell differences
            if (material_index != (materials - 1)){
                interest = layers[material_index];
                sub = 0;
                if (local_cell == (interest - 1)){
                    add = 1;
                } 
                else {
                    add = 0;
                }
            } 
            else if (material_index == (materials - 1)){
                interest = layers[material_index - 1];
                add = 0;
                if (local_cell == interest){
                    sub = 1;
                } 
                else {
                    sub = 0;
                }
            }
            */

            global_cell = change_space(gg,local_cell);
            
            A->array[global_cell][global_cell] = (2.0 / (delta * V[local_cell]) * ((D->array[material_index][gg]*D->array[material_index+add][gg]) \
                /(D->array[material_index][gg] + D->array[material_index+add][gg]) * SA[local_cell+1]) + removal->array[material_index][gg]);
            A->array[global_cell][global_cell+1] = -2.0*(D->array[material_index][gg] * D->array[material_index+add][gg]) / \
                (D->array[material_index][gg] + D->array[material_index+add][gg]) / (delta * V[local_cell]) * SA[local_cell+1];
            if (local_cell > 0){
                A->array[global_cell][global_cell-1] = -2.0*(D->array[material_index][gg] * D->array[material_index-sub][gg]) / \
                    (D->array[material_index][gg] + D->array[material_index-sub][gg]) / (delta * V[local_cell]) * SA[local_cell];
                A->array[global_cell][global_cell] += 2.0 / (delta * V[local_cell]) * ((D->array[material_index][gg]*D->array[material_index-sub][gg])/ \
                    (D->array[material_index][gg] + D->array[material_index-sub][gg]) * SA[local_cell]);
            }

            for (int gpr = 0; gpr < G; gpr++){
                if (gpr != gg){
                    prime = change_space(gpr,local_cell);
                    A->array[global_cell][prime] = -1*scatter->array[material_index][gg][gpr];
                }
            }
        }
        // Boundary Conditions
        global_cell = change_space(gg,I);
        A->array[global_cell][global_cell] = BC->array[gg][0]*0.5 + BC->array[gg][1] / delta;
        A->array[global_cell][global_cell-1] = BC->array[gg][0]*0.5 - BC->array[gg][1] / delta;
    }
}

void construct_b_lambda(void *flux, void *vector, void *birth, void *sigma_f){
    double * phi = (double *) flux;
    double * b = (double *) vector;
    double * chi = (double *) birth;
    double * fission = (double *) sigma_f;

    // Initialize indices
    int local_cell, group_in, global_y;

    for (int global_x = 0; global_x < (G*(I + 1)); global_x++){
        local_cell = global_x % (I + 1);
        if (local_cell == I){
            continue;
        }
        group_in = (int) (global_x / (I + 1));
        for (int group_out = 0; group_out < G; group_out++){
            global_y = group_out * (I + 1) + local_cell;
            b[global_x] += chi[group_in] * fission[group_out] * phi[global_y];
        }
    }
}

extern void construct_b_list(void *flux, void *vector, multi_vec *chi, multi_vec *fission, void *shape){
    double * phi = (double *) flux;
    double * b = (double *) vector;
    int * layers = (int *) shape;

    // Initialize indices
    int local_cell, group_in, global_y;
    int material_index = 0, material_cell = 0;

    for (int global_x = 0; global_x < (G*(I + 1)); global_x++){
        local_cell = global_x % (I + 1);
        if (local_cell == I){
            continue;
        }
        // This is for the material index
        if (local_cell == 0){
            material_cell = 0;            
            material_index = 0;
        } else if ((material_cell % layers[material_index]) == (layers[material_index] - 1)){
            material_cell = 0;            
            material_index += 1;
        } else {
            material_cell += 1;
        }

        group_in = (int) (global_x / (I + 1));
        for (int group_out = 0; group_out < G; group_out++){
            global_y = group_out * (I + 1) + local_cell;
            b[global_x] += (chi->array[material_index][group_in] * fission->array[material_index][group_out] * phi[global_y]);
        }
    }
}

extern void construct_b_list_fission(void *flux, void *vector, multi_mat *fission, void *shape){
    double * phi = (double *) flux;
    double * b = (double *) vector;
    int * layers = (int *) shape;

    // Initialize indices
    int local_cell, group_in, global_y;
    int material_index = 0, material_cell = 0;

    for (int global_x = 0; global_x < (G*(I + 1)); global_x++){
        local_cell = global_x % (I + 1);
        if (local_cell == I){
            continue;
        }
        // This is for the material index
        if (local_cell == 0){
            material_cell = 0;            
            material_index = 0;
        } else if ((material_cell % layers[material_index]) == (layers[material_index] - 1)){
            material_cell = 0;            
            material_index += 1;
        } else {
            material_cell += 1;
        }

        group_in = (int) (global_x / (I + 1));
        for (int group_out = 0; group_out < G; group_out++){
            global_y = group_out * (I + 1) + local_cell;
            b[global_x] += (fission->array[material_index][group_in][group_out] * phi[global_y]);
        }
    }
}