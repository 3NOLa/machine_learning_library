#pragma once
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "hashmap.h"

#define LINE_SIZE 1000

typedef struct Tensor;

typedef enum {
	INT,
	DOUBLE,
//	FLOAT,
//	CHAR,
	STRING
} ValueType;

typedef struct {
	char line_buffer[LINE_SIZE];
	char** header;
	int cols;
	int rows;
	ValueType* cols_type;
	Tensor* data;
	FILE* csv;
	HashMap** strings_map;
}csv_handler;

int int_check(char* str);
int double_check(char* str);
int char_to_int(char* str);
float  char_to_double(char* str);
int* stringCol_to_numeric(char** col, int size);
char* stringCol_to_uniq_map(char** col, int size, int* out_size);//returns an array that have any string in the col as a number(every index is the numeric value)
int* stringColAndMap_to_numeric(char** col, char** map, int map_size, int col_size);

csv_handler* csv_handler_create(char* filename);

void read_header(csv_handler* hand);
void read_next_row(csv_handler* hand);
void read_file_to_tensor(csv_handler* hand);
void print_csv_file_hand(csv_handler* hand);
void csv_handler_free(csv_handler* hand);
