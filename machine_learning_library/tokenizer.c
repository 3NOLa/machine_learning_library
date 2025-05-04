#include "tokenizer.h"
#include "stdlib.h"

char** tokeknize(const char* text, int* token_count)
{
	char** tokenizer = NULL;
	*token_count = 0;


	char* buffer = _strdup(text);
	if (!buffer) {
		fprintf(stderr, "Error: could not allocate memory for buffer copy\n");
		return NULL;
	}

	char* context = NULL;
	char* token = strtok_s(buffer, " ", &context);
	while (token)
	{
		fprintf(stderr, "%s\n", token);
		char* t = remove_punctuation(token);
		to_lowercase(t);

		char** tokenizer_check = (char**)realloc(tokenizer, sizeof(char*) * (*token_count + 1));
		if (!tokenizer_check) {
			fprintf(stderr, "Erorr: alocetying memory for tokenizer in tokeknize\n");
			return NULL;
		}

		tokenizer = tokenizer_check;
		tokenizer[(*token_count)++] = t;

		token = strtok_s(NULL, " ", &context);
	}

	free(buffer);
	return tokenizer;
}

void to_lowercase(char* str)
{
	if (!str) return;

	while (*str)
	{
		*str = tolower(*str);
		str++;
	}
}

char* remove_punctuation(const char* input)
{
	int size = strlen(input);
	char* str = (char*)malloc(sizeof(char)* (size + 1));
	if (!str) {
		fprintf(stderr, "Error: malloc failed in remove_punctuation\n");
		return NULL;
	}
		
	int j = 0;
	for (int i = 0; i < size; i++)
		if (isalnum(input[i]))
			str[(j++)] = input[i];
	str[j] = '\0';
	 
	return realloc(str, j + 1);
}