package com.example.api.controller;

import com.example.api.dto.MemberDto;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/post-api")
public class PostController {

    @RequestMapping(value = "/domain", method = RequestMethod.POST)
    public String postExample(){

        System.out.print("post API is working");
        return "Hello Post API";
    }

//    이사님한테 여쭤보기!!!!!!! 주소값이 어떻게 되는지.
    @PostMapping("/member")
//    public String postMember() {
    public String postMember(@RequestBody Map<String, Object> postData) {
//    public String postMember(@RequestBody String postData) {
        System.out.print("/member post API is working");
        StringBuilder sb = new StringBuilder();

        postData.entrySet().forEach(map -> {
            sb.append(map.getKey() + " : " + map.getValue() + "\n");
        });

        return sb.toString();
//        return postData + "ret value";
    }
//    이사님한테 여쭤보기!!!!!!! 주소값이 어떻게 되는지.
    @PostMapping("/member2")
    public String postMemberDto(@RequestBody MemberDto memberDto){
        return memberDto.toString();
    }
}
