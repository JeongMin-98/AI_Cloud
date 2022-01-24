
class SoccerPlayer(object):
    """
    네이버 부스트 코스 인공지능 기초 다지기
    2주차 OOP
    객체 생성하는 방법
    """
    def __init__(self, name : str, position : str, back_number : int):
        self.name = name
        self.position = position
        self.back_number = back_number

    def __str__(self):
        return "hello, my name is %s, My back number is %d" % \
               (self.name, self.back_number)

    def change_back_number(self, new_number):

        print("선수의 등번호를 변경합니다. : %d 에서 %d 로" % (self.back_number, new_number))
        self.back_number = new_number



abc = SoccerPlayer("son", "FW", 7)
park = SoccerPlayer("park", "WF", 13)

abc is park

abc.change_back_number(10)

print(abc)
"""
False
"""