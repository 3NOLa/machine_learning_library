#pragma once
#define MAX_VALUES 128
#define MAX_KEY_LEN 64
#define MAX_STR_LEN 32
#define MAX_LINE 512

typedef struct HashMap;

typedef enum VarType {
	TYPE_INT,
	TYPE_FLOAT,
	TYPE_ENUM
}VarType;

typedef union ConfigValues {
	int i;
	float f;
	char s[MAX_STR_LEN];
}ConfigValues;

typedef struct ConfigNode {
	char* key;
	VarType type;
	ConfigValues value[MAX_VALUES];
	int count;
	struct ConfigNode* next;
} ConfigNode;

typedef struct {
	ConfigNode** buckets;
	int size;  // Number of buckets
	int count; // Number of items
} ConfigMap;

ConfigMap* ConfigMapcreate(int size);
unsigned int Confighash_string(const char* str, int size);
int Configmap_put(ConfigMap* map, const char* key, char* valueStr);
ConfigValues* Configmap_get(ConfigMap* map, const char* key);
void Configmap_free(ConfigMap* map);

char* trim(char* str);
void parse_cfg_line(const char* line, ConfigMap* map);

int* config_values_to_int_array(ConfigValues* vals, int count);
