
# 주어진 도시는 5개로 제한.
# 방문하지 않은 도시 중에서 가장 가까운 도시를 다음 방문 도시로 정함.
import sys
import queue
MAX = sys.maxsize

city = [[MAX,99,300,100,75],
        [99,MAX,50,75,125],
        [300,50,MAX,99,124],
        [100,75,99,MAX,50],
        [75,125,124,50,MAX]]
# 도시에서 원래 도시로 가는 길은 MAX,
# ex) 0번 도시 => 2번 도시 300

start_city = 0
first_start_city = start_city
dest_city = 5
weight = 0

visited_city = [0,0,0,0,0]
# 반복 순회를 통함.

while start_city != dest_city:


    i = city[start_city].index(min(city[start_city]))
    if visited_city[i]:
        if i == first_start_city:
            break
        city[start_city][i] = MAX
    else:
        visited_city[start_city] = 1
        weight = city[start_city][i] + weight
        print("{} -> {} // {}".format(start_city,i, weight))
        start_city = i
weight = city[start_city][first_start_city] + weight
print("{} -> {} // {}".format(start_city, first_start_city, weight))




