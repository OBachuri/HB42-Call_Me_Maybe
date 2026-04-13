import json


class LLMVocabulary:
    def __init__(self, v_path: str) -> None:
        with open(v_path, "r", encoding="utf-8") as f:
            self.txt_to_id: dict[str, int] = json.load(f)
            self.id_to_txt: dict[int, str] = {v: k
                                              for k, v
                                              in self.txt_to_id.items()}
            self.int_first: set[int] = set()
            self.int_next: set[int] = set()
            self.float_next: set[int] = set()

            self.int_first = {v for k, v in self.txt_to_id.items()
                              if ((k[0].isdecimal())
                                  or ((len(k) == 1) and (k[0] in "-+"))
                                  or ((len(k) > 1)
                                      and ((
                                          (k[0] == "Ġ")
                                          and ((k[1].isdecimal())
                                               or ((k[1] in "-+")
                                                   and ((len(k) == 2)
                                                        or k[2].isdecimal()))))
                                           or ((k[0] in "-+")
                                               and (k[1].isdecimal())))
                                      ))}

            self.int_next = {v for k, v in self.txt_to_id.items()
                             if (k[0].isdecimal())}

            self.float_first = {v for k, v in self.txt_to_id.items()
                                if ((k[0].isdecimal())
                                    or ((len(k) == 1) and (k[0] in "-+."))
                                    or ((len(k) > 1)
                                        and (((k[0] == "Ġ")
                                              and ((k[1].isdecimal())
                                                   or ((k[1] in "-+.")
                                                       and ((len(k) == 2)
                                                            or k[2].isdecimal()
                                                            ))))
                                             or ((k[0] in "-+.")
                                                 and k[1].isdecimal())
                                             or ((k[0] in "-+")
                                                 and (k[1] == '.')
                                                 and ((len(k) == 2)
                                                      or k[2].isdecimal()))
                                             )
                                        ))}

            self.float_next = {v for k, v in self.txt_to_id.items()
                               if ((k[0].isdecimal())
                                   or ((k[0] in ".eE-+")
                                       and ((len(k) == 1)
                                            or (k[1].isdecimal())
                                            or ((k[0] == '.')
                                                and (k[1] in "eE")
                                                and ((len(k) == 2)
                                                     or k[2] in "1234567890+-")
                                                )
                                            or ((k[0] in "eE")
                                                and (k[1] in "+-")
                                                and ((len(k) == 2)
                                                     or k[2].isdecimal()))
                                            ))
                                   )}

            print("~~~~~~~~~~~~~~~")
            print("first-int-tokens", len(self.int_first))
            # for i in self.int_first:
            #     print(i, self.id_to_txt[i])
            print("next-int-tokens", len(self.int_next))
            print("first-float-tokens", len(self.float_first))
            # for i in self.float_first:
            #     print(i, self.id_to_txt[i])
            print("next-float-tokens", len(self.float_next))
            # for i in self.float_next:
            #     print(i, self.id_to_txt[i])

    def get_str_by_token(self, token_id: int) -> str | None:
        """ Return string representation of token """
        return self.id_to_txt.get(token_id, None)

    def get_token_by_str(self, str_: str) -> int | None:
        """ Return token id if find token for string """
        return self.txt_to_id.get(str_, None)
