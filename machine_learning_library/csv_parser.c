#include "tensor.h"
#include "csv_parser.h"

int int_check(char* str)
{
	if (!str) return 0;

	if (*str == '-' || *str == '+') str++;

	if (!*str)return 0;

	while (*str)
	{
		if (!isdigit(*str++))
			return 0;
	}

	return 1;
}

int double_check(char* str)
{
	if (!str) return 0;

	if (*str == '-' || *str == '+') str++;

	if (!*str)return 0;

	int point = 0;

	while (*str)
	{
		if (*str == '.')
		{
			if (!point)
				point = 1;
			else
				return 0;
		}
		else if (!isdigit(*str++))
			return 0;
	}

	return 1;
}


int char_to_int(char* str)
{
	if (!str) return 0;

	int sign = 1;
	int number = 0;

	if (*str == '-') {
		sign = -1;
		str++;
	}
	else if (*str == '+') {
		str++;
	}

	while (*str && isdigit(*str))
	{
		number *= 10;
		number += (*str - '0'); 
		str++;
	}

	return number * sign;
}

float  char_to_double(char* str)
{
	if (!str) return 0.0;

	float  sign = 1.0;
	float  number = 0.0;

	if (*str == '-') {
		sign = -1.0;
		str++;
	}
	else if (*str == '+') {
		str++;
	}

	while (*str && isdigit(*str))
	{
		number = number * 10.0 + (*str - '0'); 
		str++;
	}

	if (*str == '.') {
		str++;
		float  fraction = 0.1;
		while (*str && isdigit(*str))
		{
			number += (*str - '0') * fraction;
			fraction *= 0.1;
			str++;
		}
	}

	return number * sign;
}

int string_in_array(char** array,int size,char* string)
{
    if (size == 0 || !array || !string) return 0;

	for (int i = 0; i < size; i++)
	{
        if (array[i] && strcmp(array[i], string) == 0)
			return 1;
	}

	return 0;
}

char* stringCol_to_uniq_map(char** col,int size , int* out_size)//returns an array that have any string in the col as a number(every index is the numeric value)
{
    if (!col || size <= 0 || !out_size) {
        *out_size = 0;
        return NULL;
    }

	char** map = NULL;
	int k = 0;
	for (int i = 0; i < size; i++)
	{
		if (!string_in_array(map, k, col[i]))
		{
			map = (char**)realloc(map, sizeof(char*) * (k + 1));
            if (!map) {
                fprintf(stderr, "ERROR: Memory allocation failed in stringCol_to_uniq_map\n");
                *out_size = 0;
                return NULL;
            }
			map[k++] = _strdup(col[i]);
		}

	}

	*out_size = k;
	return map;
}

int* stringCol_to_numeric(char** col,int size)
{
    if (!col || size <= 0) return NULL;

    int* result = malloc(sizeof(int) * size);
    if (!result) {
        fprintf(stderr, "ERROR: Memory allocation failed in stringCol_to_numeric\n");
        return NULL;
    }

	char** map = NULL;
	int k = 0;

	for (int i = 0; i<size; i++)
	{
		if (!string_in_array(map, k, col[i]))
		{
			map = (char**)realloc(map, sizeof(char*) * (k + 1));
            if (!map) {
                fprintf(stderr, "ERROR: Memory allocation failed in stringCol_to_numeric\n");
                free(result);
                return NULL;
            }

			map[k] = _strdup(col[i]);
			result[i] = k++;
		}
		else
		{
			for (int j = 0; j < k; j++) {
				if (strcmp(col[i], map[j]) == 0) {
					result[i] = j;
					break;
				}
			}
		}

	}

	for (int i = 0; i < k; i++) {
		free(map[i]);
	}
	free(map);

	return result;
}

int* stringColAndMap_to_numeric(char** col, char** map, int map_size,int col_size) {
    if (!col || !map || map_size <= 0 || col_size <= 0) return NULL;

    int* result = malloc(sizeof(int) * col_size);
    if (!result) {
        fprintf(stderr, "ERROR: Memory allocation failed in stringColAndMap_to_numeric\n");
        return NULL;
    }

	for (int i = 0; i< col_size; i++) {
        result[i] = -1;
		for (int j = 0; j < map_size; j++) {
			if (strcmp(col[i], map[j]) == 0) {
				result[i] = j;
				break;
			}
		}
	}
	return result;
}

csv_handler* csv_handler_create(char* filename)
{
	if (!filename) {
		fprintf(stderr, "ERROR: NULL filename in csv_handler_create\n");
		return NULL;
	}

	csv_handler* h = (csv_handler*)malloc(sizeof(csv_handler));
	if (!h) {
		fprintf(stderr,"ERORR: Memory allocation failed in csv_handler_create\n");
		return NULL;
	}

    errno_t err = fopen_s(&h->csv, filename, "r");
    if (err != 0 || !h->csv) {
        fprintf(stderr, "ERROR: Could not open file '%s' in csv_handler_create\n", filename);
        free(h);
        return NULL;
    }

	h->cols = 0;
	h->rows = 0;
	h->cols_type = NULL;
	h->header = NULL;

	read_header(h);

    h->strings_map = (HashMap**)calloc(h->cols,sizeof(HashMap*));
    if (!h) {
        fprintf(stderr, "ERORR: Memory allocation for strings_map failed in csv_handler_create\n");
        return NULL;
    }

	h->data = tensor_create(2, (int[]) {h->rows, h->cols});
	if (!h->data) {
		fprintf(stderr, "ERROR: Failed to create tensor in csv_handler_create\n");
		fclose(h->csv);
		free(h->cols_type);
		for (int i = 0; i < h->cols; i++) {
			free(h->header[i]);
		}
		free(h->header);
		free(h);
		return NULL;
	}

	return h;
}

void read_header(csv_handler* hand)
{
    if (!hand || !hand->csv) return;

    if (fgets(hand->line_buffer, LINE_SIZE, hand->csv) == NULL) {
        fprintf(stderr, "ERROR: Failed to read header in read_header\n");
        return;
    }

    // Remove trailing newline if present
    size_t len = strlen(hand->line_buffer);
    if (len > 0 && (hand->line_buffer[len - 1] == '\n' || hand->line_buffer[len - 1] == '\r')) {
        hand->line_buffer[len - 1] = '\0';
    }

    hand->header = NULL;
    hand->cols = 0;
    hand->cols_type = NULL;

    // Make a copy of line_buffer because strtok modifies it
    char* buffer_copy = _strdup(hand->line_buffer);
    if (!buffer_copy) {
        fprintf(stderr, "ERROR: Memory allocation failed in read_header\n");
        return;
    }

    char* context = NULL;
    char* token = strtok_s(buffer_copy, ",", &context);
    while (token) {
        // Allocate memory for the new column type
        hand->cols_type = (ValueType*)realloc(hand->cols_type, sizeof(ValueType) * (hand->cols + 1));
        if (!hand->cols_type) {
            fprintf(stderr, "ERROR: Memory allocation failed in read_header\n");
            free(buffer_copy);
            return;
        }

        // Allocate memory for the new header entry
        hand->header = (char**)realloc(hand->header, sizeof(char*) * (hand->cols + 1));
        if (!hand->header) {
            fprintf(stderr, "ERROR: Memory allocation failed in read_header\n");
            free(buffer_copy);
            return;
        }

        // Copy token to header
        hand->header[hand->cols] = _strdup(token);

        // Analyze the column type (will be refined later)
        hand->cols_type[hand->cols] = STRING;  // Default to STRING, will check data later

        hand->cols++;
        token = strtok_s(NULL, ",", &context);
    }

    free(buffer_copy);
    hand->rows = 0;
}

void read_next_row(csv_handler* hand)
{
    if (!hand || !hand->csv) return;

    if (fgets(hand->line_buffer, LINE_SIZE, hand->csv) != NULL) {
        // Remove trailing newline if present
        size_t len = strlen(hand->line_buffer);
        if (len > 0 && (hand->line_buffer[len - 1] == '\n' || hand->line_buffer[len - 1] == '\r')) {
            hand->line_buffer[len - 1] = '\0';
        }
        hand->rows++;
    }
}

void read_file_to_tensor(csv_handler* hand)
{
    if (!hand || !hand->csv || !hand->data) return;

    // Rewind file to just after the header
    rewind(hand->csv);
    read_next_row(hand); // Skip header

    // First pass: determine types from the first data row
    if (fgets(hand->line_buffer, LINE_SIZE, hand->csv) != NULL) {
        // Remove trailing newline if present
        size_t len = strlen(hand->line_buffer);
        if (len > 0 && (hand->line_buffer[len - 1] == '\n' || hand->line_buffer[len - 1] == '\r')) {
            hand->line_buffer[len - 1] = '\0';
        }

        char* buffer_copy = _strdup(hand->line_buffer);
        if (!buffer_copy) {
            fprintf(stderr, "ERROR: Memory allocation failed in read_file_to_tensor\n");
            return;
        }

        char* context = NULL;
        char* token = strtok_s(buffer_copy, ",", &context);
        int col_idx = 0;

        while (token && col_idx < hand->cols) {
            if (int_check(token)) {
                hand->cols_type[col_idx] = INT;
            }
            else if (double_check(token)) {
                hand->cols_type[col_idx] = DOUBLE;
            }
            else {
                hand->cols_type[col_idx] = STRING;
                hand->strings_map[col_idx] = hashmap_create(1000);
            }
            col_idx++;
            token = strtok_s(NULL, ",", &context);
        }

        free(buffer_copy);
    }

    // Rewind again to begin processing all data
    rewind(hand->csv);
    read_next_row(hand); // Skip header

    // Count rows to determine tensor size
    int row_count = 0;
    char line[LINE_SIZE];
    while (fgets(line, LINE_SIZE, hand->csv)) {
        row_count++;
    }

    // Resize tensor to accommodate all rows
    tensor_free(hand->data);
    hand->data = tensor_create(2, (int[]) { row_count, hand->cols });
    if (!hand->data) {
        fprintf(stderr, "ERROR: Failed to create tensor in read_file_to_tensor\n");
        return;
    }

    // Rewind again to read all data
    rewind(hand->csv);
    read_next_row(hand); // Skip header
    int* next_indices = (int*)calloc(hand->cols, sizeof(int)); 

    // Process all rows of data
    int row_idx = 0;
    while (fgets(hand->line_buffer, LINE_SIZE, hand->csv)) {
        // Remove trailing newline if present
        size_t len = strlen(hand->line_buffer);
        if (len > 0 && (hand->line_buffer[len - 1] == '\n' || hand->line_buffer[len - 1] == '\r')) {
            hand->line_buffer[len - 1] = '\0';
        }

        char* buffer_copy = _strdup(hand->line_buffer);
        if (!buffer_copy) {
            fprintf(stderr, "ERROR: Memory allocation failed in read_file_to_tensor\n");
            return;
        }

        char* context = NULL;
        char* token = strtok_s(buffer_copy, ",", &context);
        int col_idx = 0;

        while (token && col_idx < hand->cols) {
            float  value = 0.0;

            if (hand->cols_type[col_idx] == INT) {
                value = (float )char_to_int(token);
            }
            else if (hand->cols_type[col_idx] == DOUBLE) {
                value = char_to_double(token);
            }
            else if (hand->cols_type[col_idx] == STRING) {
                int found = hashmap_get(hand->strings_map[col_idx], token);
                if (found == -1) {
                    value = next_indices[col_idx]++;
                    hashmap_put(hand->strings_map[col_idx], token, value);
                }
                else {
                    value = hashmap_get(hand->strings_map[col_idx], token);
                }
            }
            else {
                value = 0.0;
            }

            tensor_set(hand->data, (int[]) { row_idx, col_idx }, value);
            col_idx++;
            token = strtok_s(NULL, ",", &context);
        }

        free(buffer_copy);
        row_idx++;
    }

    hand->rows = row_count;
}

void print_csv_file_hand(csv_handler* hand)
{
    if (!hand) {
        fprintf(stderr, "Erorr: no handler in print_csv_file_hand\n");
        return;
    }

    fprintf(stderr, "Printing csv file:\n");
    fprintf(stderr, "cols: %d, rows: %d\n",hand->cols,hand->rows);

    fprintf(stderr,"Header:\t\t");
    for (int i = 0; i < hand->cols; i++)
    {
        fprintf(stderr, "%s\t", hand->header[i]);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "Types:\t\t");
    for (int i = 0; i < hand->cols; i++)
    {
        const char* type_str;
        switch (hand->cols_type[i]) {
        case INT:
            type_str = "INT";
            break;
        case DOUBLE:
            type_str = "DOUBLE";
            break;
        case STRING:
            type_str = "STRING";
            break;
        default:
            type_str = "UNKNOWN";
        }
        fprintf(stderr, "%s\t", type_str);
    }
    fprintf(stderr, "\n");

    // Print data preview (up to 5 rows)
    fprintf(stderr, "Data preview:\n");
    int rows_preview = (5 > hand->rows) ? hand->rows : 5;

    for (int r = 0; r < rows_preview; r++)
    {
        fprintf(stderr, "Row %d:\t\t", r);
        for (int c = 0; c < hand->cols; c++)
        {
            if (hand->data && r < hand->data->shape[0] && c < hand->data->shape[1]) {
                float  value = tensor_get_element(hand->data, (int[]) { r, c });

                switch (hand->cols_type[c]) {
                case INT:
                    fprintf(stderr, "%d\t\t", (int)value);
                    break;
                case DOUBLE:
                    fprintf(stderr, "%.3f\t\t", value);
                    break;
                case STRING:
                    fprintf(stderr, "%.0f\t\t", value);
                    break;
                }
            }
            else {
                fprintf(stderr, "N/A\t\t\t");
            }
        }
        fprintf(stderr, "\n");
    }
}

void csv_handler_free(csv_handler* hand)
{
    if (!hand) return;

    if (hand->csv) {
        fclose(hand->csv);
    }

    if (hand->header) {
        for (int i = 0; i < hand->cols; i++) {
            free(hand->header[i]);
        }
        free(hand->header);
    }

    if (hand->cols_type) {
        free(hand->cols_type);
    }

    if (hand->data) {
        tensor_free(hand->data);
    }

    free(hand);
}