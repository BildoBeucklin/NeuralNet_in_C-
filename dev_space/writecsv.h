#ifndef _WRITECSV_H
#define _WRITECSV_H

static const char weights_path[] = "./data/weights.csv";
static const char biases_path[] = "./data/biases.csv";

void writecsv(struct net* net, struct stp* stp);

#endif