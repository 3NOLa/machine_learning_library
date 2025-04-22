#include "hashmap.h"

// Create a new hashmap
HashMap* hashmap_create(int size) {
    HashMap* map = (HashMap*)malloc(sizeof(HashMap));
    if (!map) return NULL;

    map->size = size;
    map->count = 0;
    map->buckets = (HashNode**)calloc(size, sizeof(HashNode*));
    if (!map->buckets) {
        free(map);
        return NULL;
    }

    return map;
}

// Simple hash function for strings
unsigned int hash_string(const char* str, int size) {
    unsigned int hash = 0;
    while (*str) {
        hash = (hash * 31) + (*str++);
    }
    return hash % size;
}

// Insert a key-value pair into the hashmap
int hashmap_put(HashMap* map, const char* key, int value) {
    if (!map || !key) return 0;

    unsigned int index = hash_string(key, map->size);
    HashNode* current = map->buckets[index];

    // Check if key already exists
    while (current) {
        if (strcmp(current->key, key) == 0) {
            current->value = value;
            return 1;
        }
        current = current->next;
    }

    // Key doesn't exist, create new node
    HashNode* new_node = (HashNode*)malloc(sizeof(HashNode));
    if (!new_node) return 0;

    new_node->key = _strdup(key);
    if (!new_node->key) {
        free(new_node);
        return 0;
    }

    new_node->value = value;
    new_node->next = map->buckets[index];
    map->buckets[index] = new_node;
    map->count++;

    return 1;
}

// Get value for a key, or -1 if not found
int hashmap_get(HashMap* map, const char* key) {
    if (!map || !key) return -1;

    unsigned int index = hash_string(key, map->size);
    HashNode* current = map->buckets[index];

    while (current) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next;
    }

    return -1;  // Not found
}

// Free the hashmap
void hashmap_free(HashMap* map) {
    if (!map) return;

    for (int i = 0; i < map->size; i++) {
        HashNode* current = map->buckets[i];
        while (current) {
            HashNode* temp = current;
            current = current->next;
            free(temp->key);
            free(temp);
        }
    }

    free(map->buckets);
    free(map);
}