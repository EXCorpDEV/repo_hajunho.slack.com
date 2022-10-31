package com.example.api.controller;


import java.util.Map;

import com.example.api.dto.MemberDto;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/get-api")
public class GetController {

    @RequestMapping(value = "/hello",method = RequestMethod.GET)
    public String getHello(){
        return "Hello World";
    }
    @GetMapping("/name")
    public String getName(){
        return "Flatrue";
    }
//    <예시1번>
//                                 원하는 값을 넣어준다.
    @GetMapping("/variable1/{variable}")
//    @PathVariable -> 간단히 전달할 때 주로 사용
    public String getVariable1(@PathVariable String variable){
        return variable;
    }
//    <예시2>
    @GetMapping("/test1/{name}/{name2}")
    public String Test1(@PathVariable String name,@PathVariable String name2){
        return name + "와"+ name2;
    }

//    변수명을 다르게 성정하고 싶을때 ex)variable를 설정하고 얘의 이름을 바꿔준다고(var) 생각하면된다.
//    // http://localhost:8080/api/v1/get-api/variable2/{String}
    @GetMapping(value = "/variable2/{variable}")
    public String getVariable2(@PathVariable("variable") String var) {
        return var;
    }

    @GetMapping("/request1")
    public String getRequestParam1(
            @RequestParam String name,
            @RequestParam String email,
            @RequestParam String organization){
        return name + " " + email + " " + organization;

    }

    @GetMapping("/request2")
//                                      값이 어떻게 들어올지 모를때 사용하면 좋다.
    public String getRequestParam2(@RequestParam Map<String, String> param) {
        StringBuilder sb = new StringBuilder();

        param.entrySet().forEach(map -> {
            sb.append(map.getKey() + " : " + map.getValue() + "\n");
        });

        return sb.toString();
    }

//    http://localhost:8080/get/request3?name=value1&email=value2&organization=value3
//    DTO에서 가져온거
    @GetMapping("/request3")
    public String getRequestParam3(MemberDto memberDto){
// return memberDto.getName() + " " + memberDto.getEmail() + " " + memberDto.getOrganization();
        return memberDto.toString();
    }


}
