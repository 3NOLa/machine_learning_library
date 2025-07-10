#include "cfgparse.h"
#include <ctype.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

ConfigMap* ConfigMapcreate(int size) {
    ConfigMap* map = (ConfigMap*)malloc(sizeof(ConfigMap));
    if (!map) return NULL;

    map->size = size;
    map->count = 0;
    map->buckets = (ConfigNode**)calloc(size, sizeof(ConfigNode*));
    if (!map->buckets) {
        free(map);
        return NULL;
    }

    return map;
}

unsigned int Confighash_string(const char* str, int size) {
    unsigned int hash = 0;
    while (*str) {
        hash = (hash * 31) + (*str++);
    }
    return hash % size;
}

int Configmap_put(ConfigMap* map, const char* key, char* valueStr) {
    if (!map || !key) return 0;

    unsigned int index = Confighash_string(key, map->size);
    ConfigNode* current = map->buckets[index];

    while (current) {
        if (strcmp(current->key, key) == 0) {
            int idx = current->count++;

            switch (current->type) {
            case TYPE_INT:
                current->value[idx].i = atoi(valueStr);
                break;
            case TYPE_FLOAT:
                current->value[idx].f = atof(valueStr);
                break;
            case TYPE_ENUM:
                strncpy_s(current->value[idx].s, MAX_STR_LEN, valueStr, _TRUNCATE);
                current->value[idx].s[MAX_STR_LEN - 1] = '\0';
                break;
            }
            return 1;
        }
        current = current->next;
    }

    ConfigNode* new_node = (ConfigNode*)malloc(sizeof(ConfigNode));
    if (!new_node) return 0;

    new_node->key = _strdup(key);
    if (!new_node->key) {
        free(new_node);
        return 0;
    }

    if (strchr(valueStr, '.')) {
        new_node->type = TYPE_FLOAT;
        new_node->value[0].f = atof(valueStr);
    }
    else if (isdigit(valueStr[0]) || valueStr[0] == '-') {
        new_node->type = TYPE_INT;
        new_node->value[0].i = atoi(valueStr);
    }
    else {
        new_node->type = TYPE_ENUM;
        strncpy_s(new_node->value[0].s, MAX_STR_LEN, valueStr, _TRUNCATE);
    }
    new_node->count = 1;
    new_node->next = map->buckets[index];
    map->buckets[index] = new_node;
    map->count++;

    return 1;
}

ConfigValues* Configmap_get(ConfigMap* map, const char* key) {
    if (!map || !key) return NULL;

    unsigned int index = Confighash_string(key, map->size);
    ConfigNode* current = map->buckets[index];

    while (current) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next;
    }

    return -1;  // Not found
}

void Configmap_free(ConfigMap* map) {
    if (!map) return;

    for (int i = 0; i < map->size; i++) {
        ConfigNode* current = map->buckets[i];
        while (current) {
            ConfigNode* temp = current;
            current = current->next;
            free(temp->key);
            free(temp);
        }
    }

    free(map->buckets);
    free(map);
}

char* trim(char* str) {
    while (isspace(*str)) str++;
    char* end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) *end-- = '\0';
    return str;
}

void parse_cfg_line(const char* line, ConfigMap* map) {
    char key[MAX_KEY_LEN];
    char valueStr[256];

    const char* eq = strchr(line, '=');
    if (!eq) return;

    size_t keyLen = eq - line;
    if (keyLen >= MAX_KEY_LEN) keyLen = MAX_KEY_LEN - 1;
    strncpy_s(key, MAX_KEY_LEN, line, keyLen);
    key[keyLen] = '\0';

    strcpy_s(valueStr,MAX_STR_LEN , eq + 1);

    char* context = NULL;
    char* token = strtok_s(valueStr, ",", &context);
    while (token) {
        Configmap_put(map, trim(key), trim(token));
        token = strtok_s(NULL, ",", &context);
    }
}

int* config_values_to_int_array(ConfigValues* vals, int count) {
    if (!vals || count <= 0) return NULL;

    int* result = malloc(sizeof(int) * count);
    if (!result) return NULL;

    for (int i = 0; i < count; i++) {
        result[i] = vals[i].i;
    }

    return result;
}

