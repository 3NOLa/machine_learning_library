#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef enum{
	map,
	bigmap,
	circle,
	box,
	race
}envoType;

typedef union {
	int map[3][3];
	int bigmap[4][4];

}envoBuilds;

typedef struct {
	envoType type;
	envoBuilds build;
}enviroment;


typedef struct {

}agent;