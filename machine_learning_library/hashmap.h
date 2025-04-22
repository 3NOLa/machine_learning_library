#pragma once
#include <stdlib.h>
#include <string.h>

typedef struct HashNode {
    char* key;
    int value;
    struct HashNode* next;
} HashNode;

typedef struct {
    HashNode** buckets;
    int size;  // Number of buckets
    int count; // Number of items
} HashMap;

HashMap* hashmap_create(int size);
unsigned int hash_string(const char* str, int size);
int hashmap_put(HashMap* map, const char* key, int value);
int hashmap_get(HashMap* map, const char* key);
void hashmap_free(HashMap* map);