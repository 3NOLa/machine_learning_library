#pragma once
#include <string.h>
#include <ctype.h>
#include <stdio.h>

char** tokeknize(const char* text, int* token_count);
void to_lowercase(char* str);
char* remove_punctuation(const char* input);