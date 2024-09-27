for i in {1..9}; do cp ./test_rw_SQ_mo_0.py ./test_rw_SQ_mo_${i}.py ; done
for i in {1..9}; do sed -i "21s/0/${i}/g" ./test_rw_SQ_mo_${i}.py; done
for i in {1..9}; do sed -i "54s/0/${i}/g" ./test_rw_SQ_mo_${i}.py; done
for i in {0..9}; do pysub_nchc_n2 test_rw_SQ_mo_${i}; done
