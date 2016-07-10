#ifndef BIO_UTILS_H
#define BIO_UTILS_H

#include <iostream>
#include <cstring>
#include <random>
#include <unordered_map>

std::unordered_map<char, std::vector<char> > ambig_table = {
        {'R', {'A', 'G'}},
        {'Y', {'C', 'T'}},
        {'S', {'G', 'C'}},
        {'W', {'A', 'T'}},
        {'K', {'G', 'T'}},
        {'M', {'A', 'C'}},
        {'B', {'C', 'G', 'T'}},
        {'D', {'A', 'G', 'T'}},
        {'H', {'A', 'C', 'T'}},
        {'V', {'A', 'C', 'G'}},
        {'N', {'A', 'C', 'G', 'T'}}
};

void replaceAmbigs(std::string &str) {
    for(int i = 0; i < str.length(); ++i) {
        if(ambig_table.count(str[i])) {
            int rand_idx = rand() % ambig_table[str[i]].size();
            str[i] = ambig_table[str[i]][rand_idx];
        }
    }
}



#endif // BIO_UTILS_H
