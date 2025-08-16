#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <algorithm>
#include "Int.h"

#pragma pack(push,1)
struct DpItem {
    uint64_t fp;
    uint8_t  priv[32];
};
#pragma pack(pop)

struct RangeSeg {
    Int start;
    Int length;
};

static std::vector<RangeSeg>
splitRange(const Int &A, const Int &total, unsigned parts) {
    std::vector<RangeSeg> seg(parts);
    Int chunk(total), lenLast(total);
    { Int div((uint64_t)parts); chunk.Div(&div,nullptr); }
    if(parts>1){
        Int tmp(chunk), m((uint64_t)(parts-1));
        tmp.Mult(&m);
        lenLast.Sub(&tmp);
    }
    for(unsigned i=0;i<parts;++i){
        seg[i].start = A;
        if(i){
            Int off(chunk), k((uint64_t)i);
            off.Mult(&k);
            seg[i].start.Add(&off);
        }
        seg[i].length = (i+1==parts? lenLast: chunk);
    }
    return seg;
}

static Int parseInt(const std::string &s) {
    Int x;
    if(s.rfind("0x",0)==0 || s.find_first_of("ABCDEFabcdef")!=std::string::npos)
        x.SetBase16((char*)s.c_str());
    else
        x.SetBase10((char*)s.c_str());
    return x;
}

int main(int argc, char **argv) {
    unsigned segments=0;
    std::string range_s, dp_path="DP.bin";
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        if(a=="--segments"&&i+1<argc) segments=std::stoul(argv[++i]);
        else if(a=="--range"&&i+1<argc) range_s=argv[++i];
        else if(a=="--dp-file"&&i+1<argc) dp_path=argv[++i];
        else { std::cerr<<"Unknown arg: "<<a<<"\n"; return 1; }
    }
    if(segments<1){ std::cerr<<"Error: need --segments N\n"; return 1; }
    auto pos=range_s.find(':');
    if(pos==std::string::npos){ std::cerr<<"Error: need --range A:B\n"; return 1; }

    Int A=parseInt(range_s.substr(0,pos));
    Int B=parseInt(range_s.substr(pos+1));
    Int total(B); total.Sub(&const_cast<Int&>(A));
    auto segs=splitRange(A,total,segments);

    std::vector<uint64_t> cnt(segments,0);
    std::vector<std::vector<Int>> privs(segments);
    uint64_t total_items=0;
    std::ifstream in(dp_path,std::ios::binary);
    if(!in){ std::cerr<<"Cannot open DP-file: "<<dp_path<<"\n"; return 1; }
    DpItem item;
    while(in.read((char*)&item,sizeof(item))){
        ++total_items;
        Int x; x.Set32Bytes(item.priv);
        for(unsigned i=0;i<segments;++i){
            if(x.IsGreaterOrEqual(&segs[i].start)){
                Int e(segs[i].start); e.Add(&segs[i].length);
                if(x.IsLower(&e)){
                    cnt[i]++; privs[i].push_back(x);
                    break;
                }
            }
        }
    }
    in.close();

    uint64_t empties=0, minc=UINT64_MAX, maxc=0;
    double sum=0,sum2=0;
    for(auto c:cnt){
        if(c==0) empties++;
        minc = std::min(minc,c);
        maxc = std::max(maxc,c);
        sum  += double(c);
        sum2 += double(c)*double(c);
    }
    double mean = sum/segments;
    double var  = sum2/segments - mean*mean;

    std::vector<std::string> bitl(segments), minG(segments,"-"), p10G(segments,"-"),
                             medG(segments,"-"), p90G(segments,"-"), avgG(segments,"-"),
                             maxG(segments,"-"), dens(segments,"-"), recK(segments,"-"),
                             skew(segments,"-");
    for(unsigned i=0;i<segments;++i){
        bitl[i]=std::to_string(segs[i].length.GetBitLength());
        auto &v=privs[i];
        if(v.size()>1){
            std::sort(v.begin(),v.end(),[&](auto &a,auto &b){
                return const_cast<Int&>(a).IsLower(const_cast<Int*>(&b));
            });
            size_t n=v.size()-1;
            std::vector<double> gd(n);
            for(size_t j=0;j<n;++j){
                Int d=v[j+1]; d.Sub(&const_cast<Int&>(v[j]));
                gd[j]=d.ToDouble();
            }
            std::sort(gd.begin(),gd.end());
            auto pct=[&](double p){ return gd[size_t(std::floor(p*(n-1)))]; };
            double dmin=gd.front(), d10=pct(0.10), d50=pct(0.50),
                   d90=pct(0.90), dmax=gd.back();
            double s=0; for(double x:gd) s+=x;
            double davg=s/n;
            double ss=0; for(double x:gd) ss+=(x-davg)*(x-davg);
            double dstd=sqrt(ss/n);
            double densv = cnt[i]/segs[i].length.ToDouble();
            double krecd = 0.5*log2(davg);
            double dskew = (dmax-davg)/(davg-dmin);
            minG[i] = std::to_string(int64_t(dmin));
            p10G[i]= std::to_string(int64_t(d10));
            medG[i]= std::to_string(int64_t(d50));
            p90G[i]= std::to_string(int64_t(d90));
            avgG[i] = std::to_string(int64_t(davg));
            maxG[i] = std::to_string(int64_t(dmax));
            dens[i] = std::to_string(densv);
            recK[i] = std::to_string(int(floor(krecd)));
            skew[i] = std::to_string(dskew);
        }
    }

    const int W_S=7, W_C=10, W_P=9, W_B=5,
              W_G=10, W_D=12, W_R=6, W_Sk=8;

    std::ostringstream hdr;
    hdr << std::setw(W_S) << "Seg"   << " |"
        << std::setw(W_C) << "Count" << " |"
        << std::setw(W_P) << "%tot"  << "  |"
        << std::setw(W_B) << "bits"  << " |"
        << std::setw(W_G) << "minGap"<< " |"
        << std::setw(W_G) << "p10Gap"<< " |"
        << std::setw(W_G) << "medGap"<< " |"
        << std::setw(W_G) << "p90Gap"<< " |"
        << std::setw(W_G) << "avgGap"<< " |"
        << std::setw(W_G) << "maxGap"<< "    |"
        << std::setw(W_D) << "density"<< " |"
        << std::setw(W_R) << "k_rec" << " |"
        << std::setw(W_Sk)<< "skew";
    std::string header = hdr.str();

    std::cout << "\nDP traps distribution over " << segments << " segments:\n\n";
    std::cout << header << "\n";
    std::cout << std::string(header.size(), '-') << "\n";

    for (unsigned i = 0; i < segments; ++i) {
        double pct = total_items ? 100.0 * cnt[i] / double(total_items) : 0.0;
        std::cout
            << std::setw(W_S) << i      << " |"
            << std::setw(W_C) << cnt[i] << " |"
            << std::setw(W_P) << std::fixed << std::setprecision(2) << pct << "% |"
            << std::setw(W_B) << bitl[i]<< " |"
            << std::setw(W_G) << minG[i]<< " |"
            << std::setw(W_G) << p10G[i]<< " |"
            << std::setw(W_G) << medG[i]<< " |"
            << std::setw(W_G) << p90G[i]<< " |"
            << std::setw(W_G) << avgG[i]<< " |"
            << std::setw(W_G) << maxG[i]<< " |"
            << std::setw(W_D) << dens[i]<< " |"
            << std::setw(W_R) << recK[i]<< " |"
            << std::setw(W_Sk)<< skew[i] << "\n";
    }

    std::cout << "\nEmpty segments: " << empties
              << "   Min/Max/Mean/Var count: "
              << minc <<"/"<< maxc <<"/"
              << std::fixed<<std::setprecision(2)<< mean <<"/"<< var << "\n";

    return 0;
}
