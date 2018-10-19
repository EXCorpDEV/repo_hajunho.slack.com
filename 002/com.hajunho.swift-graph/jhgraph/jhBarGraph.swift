//
//  jhBarGraph.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhBarGraph : jhPanel {
    override func drawDatas() {
        var pointCloud = Array<CGPoint>()
        var fx, fy : CGFloat
        
        var x : Int = 0
        for y in data.mValuesOfDatas {
            x += 1
            fx = CGFloat(x)*data.axisDistance
            fy = CGFloat(y)*data.mVerticalRatioToDraw_view + self.data.mMargin
            drawEllipse(fx, fy, 2, 2, thickness: 2, UIColor.blue.cgColor)
            pointCloud.append(CGPoint.init(x: getX(fx+self.data.mMargin)!, y: getY(fy)!))
        }
        
        mContext?.move(to: CGPoint.init(x: 0, y: 0))
        mContext?.setStrokeColorSpace(CGColorSpaceCreateDeviceRGB())
        mContext?.setStrokeColor(UIColor.blue.cgColor)
        mContext?.setLineWidth(1.0)
        mContext?.addLines(between: pointCloud)
        mContext?.strokePath()
        
        for x in pointCloud {
            mContext?.setFillColor(jhColor(red: 184, green: 70, blue: 201, alpha: 0.5))
            mContext?.setStrokeColor(jhColor(red: 184, green: 70, blue: 201, alpha: 1.0))
            mContext?.setLineWidth(1)
            
            let rectangle = CGRect(x: x.x-5, y: x.y, width: 10, height: -(x.y-getY(self.data.mMargin)!)) //TODO: 좌표 계산 부분 한 곳으로 몰기.
            mContext?.addRect(rectangle)
            mContext?.drawPath(using: .fillStroke)
        }
    }
}

