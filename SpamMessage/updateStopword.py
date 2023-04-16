if __name__ == "__main__":
    with open("new_stopwords.txt",mode="w+",encoding='utf-8') as new:
        scu = open("scu_stopwords.txt")
        hit = open("hit_stopwords.txt")
        bdu = open("baidu_stopwords.txt")
        scu_list = scu.readlines()
        hit_list = hit.readlines()
        bdu_list = bdu.readlines()
        new.writelines(set(scu_list+hit_list+bdu_list))
        scu.close()
        hit.close()
        bdu.close()