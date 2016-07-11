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

template <typename T>
void load_kmer_array(const std::string &r1, const std::string &r2, const T &f, const int k) {
    int m = 0;
    for(; m < r.length() - k + 1; ++m) {
        for(int n = 0; n < k; ++n) {
            f[(m * k) + n] = reinterpret_cast<unsigned char &>(r1[m + n]);
        }
    }
    for(int t = 0; t < r2.length() - k + 1; ++m, ++t) {
        for(int n = 0; n < k; ++n) {
            f[(m * k) + n] = reinterpret_cast<unsigned char &>(r2[t + n]);
        }
    }
}



#endif // BIO_UTILS_H
